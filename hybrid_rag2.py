# %%
"""
# Hybrid RAG Implementation
Combines Multi-Hop RAG, Corrective RAG (CRAG), and Self-RAG to create a robust, dynamic Business Intelligence assistant.
"""

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.utilities import SQLDatabase
from google.cloud import bigquery
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
import langgraph.graph as lg
import os
from tavily import TavilyClient
import pandas as pd
from tools.nl2sql_tools import write_query, execute_query, summarize_result

# %%
load_dotenv()
client = bigquery.Client()
project_id = client.project
db = SQLDatabase.from_uri(f"bigquery://{project_id}")

print("loaded database")

# LLMs
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("loaded llms")

#%% 
class GraphState(TypedDict):
    question: str
    initial_answer: str
    generation: str
    documents: List[str]
    subquestions: List[str]
    all_subquestions: List[str]
    iteration: int
    hallucination: str
    answer: str
    came_from_subq: bool
    is_multi_hop: bool
    context_iterations: int
    reflection: str
    causal_explanation: str
    current_phase: str  # New field to track where we are in the process

# Vector store
embedding = OpenAIEmbeddings()
documents = [Document(page_content="Initial placeholder document.")]
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()
# %%

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")

structured_grader = llm.with_structured_output(GradeDocuments)

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="'yes' if it answers the question with factual information from the SQL results")
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are evaluating if a SQL query result answers the user's question. Consider it a 'yes' if it provides factual information from the database, even if the numbers are small or the result is simple. Only say 'no' if it completely fails to address the question or contains no relevant information."),
    ("human", "Question: {question}\nGeneration: {generation}")
])
answer_grader = answer_prompt | structured_grader

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="'no' if the answer only contains information from the SQL results, 'yes' if it makes up information not in the results")
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are evaluating if an answer contains information not present in the SQL results. Say 'no' if the answer only contains information that could come from the SQL results, even if the numbers are small. Say 'yes' only if it makes up information not present in the results."),
    ("human", "Documents: {documents}\nGeneration: {generation}")
])
hallucination_grader = hallucination_prompt | structured_grader

class GradeCausalExplanation(BaseModel):
    binary_score: str = Field(description="'yes' if it provides good causal explanation with supporting evidence, 'no' if not")
    reasoning: str = Field(description="Brief explanation of the grading decision")

causal_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are evaluating if an answer provides good causal explanation.
    Consider it 'yes' if it:
    1. Explains WHY this is the result (not just what the result is)
    2. Provides supporting evidence from the data
    3. Makes logical connections between different pieces of information
    4. Gives business context for the findings
    
    Only say 'no' if it's purely descriptive or lacks supporting evidence."""),
    ("human", "Question: {question}\nAnswer: {generation}")
])

causal_grader = llm.with_structured_output(GradeCausalExplanation)
# %%
def analyze_initial_sql_node(state):
    """Get initial answer from SQL"""
    print("\n[INITIAL_QUERY] Starting initial query analysis")
    state["current_phase"] = "initial_query"
    
    try:
        sql_query = write_query(state["question"])
        print(f"[INITIAL_QUERY] Generated SQL: {sql_query}")
        
        result = execute_query(sql_query)
        print("[INITIAL_QUERY] Executed SQL query")
        
        summary = summarize_result(sql_query, result, state["question"])
        print(f"[INITIAL_QUERY] Generated summary: {summary}")
        
        # Add to vector store
        initial_doc = Document(page_content=summary)
        vectorstore.add_documents([initial_doc])
        
        return {
            **state,
            "initial_answer": summary,
            "documents": [summary],
            "generation": summary,
            "iteration": 0,
            "context_iterations": 0
        }
    except Exception as e:
        print(f"[INITIAL_QUERY] Error: {str(e)}")
        raise

def grade(state):
    """Enhanced grading that considers both factual accuracy and causal explanation"""
    print(f"\n[GRADE] Starting grading for question: {state['question']}")
    print(f"[GRADE] Current generation: {state['generation'][:200]}...")
    
    # Grade factual accuracy
    hallucination_score = hallucination_grader.invoke({
        "documents": state["documents"],
        "generation": state["generation"]
    }).binary_score
    print(f"[GRADE] Hallucination score: {hallucination_score}")
    
    answer_score = answer_grader.invoke({
        "question": state["question"],
        "generation": state["generation"]
    }).binary_score
    print(f"[GRADE] Answer score: {answer_score}")
    
    # Grade causal explanation if we're in multi-hop phase
    if state.get("is_multi_hop", False):
        # Create the prompt for causal evaluation
        causal_evaluation_prompt = f"""
        Evaluate if this answer provides good causal explanation:
        Question: {state['question']}
        Answer: {state['generation']}
        
        Consider it 'yes' if it:
        1. Explains WHY this is the result (not just what the result is)
        2. Provides supporting evidence from the data
        3. Makes logical connections between different pieces of information
        4. Gives business context for the findings
        
        Only say 'no' if it's purely descriptive or lacks supporting evidence.
        """
        
        causal_result = causal_grader.invoke({
            "question": state['question'],
            "generation": causal_evaluation_prompt
        })
        causal_score = causal_result.binary_score
        print(f"[GRADE] Causal explanation score: {causal_score}")
    else:
        causal_score = "no"
    
    return {
        **state,
        "hallucination": hallucination_score,
        "answer": answer_score,
        "causal_explanation": causal_score
    }

def retrieve_previous_results(state):
    """Retrieve and analyze previous results to inform next generation"""
    print(f"\n[RETRIEVE] Analyzing previous results for: {state['question']}")
    
    # Get all previous documents
    all_docs = state["documents"]
    
    # Create a prompt to analyze previous attempts
    analysis_prompt = f"""
    Analyze these previous attempts to answer the question: "{state['question']}"
    
    Previous attempts:
    {all_docs}
    
    What was missing or could be improved in these attempts?
    What additional information should we look for?
    """
    
    analysis = llm.invoke(analysis_prompt).content
    print(f"[RETRIEVE] Analysis: {analysis}")
    
    return {
        **state,
        "analysis": analysis
    }

def retrieve_for_expansion(state):
    """Enhanced reflection and multi-hop question generation focused on database-answerable questions"""
    print(f"\n[RETRIEVE_EXPANSION] Planning multi-hop expansion for: {state['question']}")
    
    reflection_prompt = f"""
    Original question: {state['question']}
    Initial answer: {state['documents'][0]}
    
    Current answer quality:
    - Factual accuracy: {state['answer']}
    - Causal explanation: {state.get('causal_explanation', 'no')}
    
    Understanding that we're answering: "{state['question']}"
    Analyze what additional information we could get FROM OUR DATABASE to provide more context and explanation for this specific question.
    
    Focus on measurable, internal metrics such as:
    1. Sales patterns (time-based trends, seasonal effects)
    2. Customer behavior (purchase frequency, basket analysis)
    3. Product performance (units sold, revenue, returns)
    4. Regional or store-level analysis
    5. Price points and discount impacts
    
    DO NOT suggest external factors we can't measure like:
    - Competitor actions
    - Market research
    - Customer satisfaction surveys
    - Brand perception
    - External market conditions
    
    Provide a structured analysis of what additional DATABASE QUERIES could help explain these specific results.
    """
    
    analysis = llm.invoke(reflection_prompt).content
    print(f"[RETRIEVE_EXPANSION] Analysis: {analysis}")
    
    # Generate specific follow-up questions
    subq_prompt = f"""
    Original question: {state['question']}
    Initial answer: {state['documents'][0]}
    
    Based on this analysis:
    {analysis}
    
    Generate 2-3 specific questions that:
    1. Can be answered using SQL queries on our sales/product/customer database
    2. Focus on internal metrics and measurable data points
    3. Will help provide context and explanation for the initial answer
    4. Are relevant to the original question: "{state['question']}"
    
    Format: Output ONLY the questions, one per line, no numbering or empty lines.
    Each question should be specific and SQL-answerable, focusing on metrics we can get from our database.

    For example, if the original question was about total sales in January 2024:
    Good examples:
    - How do these January 2024 sales compare to previous months?
    - What were the top 5 product categories contributing to January sales?
    - What was the daily sales distribution within January 2024?

    Bad examples (DO NOT USE):
    - Why did customers buy more in January?
    - How did market conditions affect sales?
    - What was customer satisfaction like?
    """
    
    subquestions = llm.invoke(subq_prompt).content.split("\n")
    subquestions = [q.strip() for q in subquestions if q.strip()]
    print(f"[RETRIEVE_EXPANSION] Generated subquestions: {subquestions}")
    
    return {
        **state,
        "is_multi_hop": True,
        "analysis": analysis,
        "subquestions": subquestions,
        "context_iterations": state.get("context_iterations", 0) + 1
    }

def expand_knowledge(state):
    """Execute multi-hop queries based on subquestions and synthesize all results"""
    print(f"\n[EXPAND] Executing multi-hop queries for iteration {state['context_iterations']}")
    
    if not state["subquestions"]:
        print("[EXPAND] No subquestions remaining")
        return state
    
    # Process all subquestions first
    all_new_information = []
    print(f"[EXPAND] Processing {len(state['subquestions'])} subquestions")
    
    try:
        # Gather information for ALL subquestions
        for question in state["subquestions"]:
            print(f"[EXPAND] Executing query for: {question}")
            sql_query = write_query(question)
            result = execute_query(sql_query)
            summary = summarize_result(sql_query, result, question)
            all_new_information.append({
                "question": question,
                "summary": summary
            })
            print(f"[EXPAND] Got results for: {question}")
        
        # Create a comprehensive synthesis using all gathered information
        synthesis_prompt = (
            f"Original question: {state['question']}\n"
            f"Initial answer: {state['documents'][0]}\n\n"
            f"Previous analysis identified these gaps in our explanation:\n"
            f"{state.get('analysis', 'No previous analysis available')}\n\n"
            f"Additional information gathered:\n"
            + "\n".join([f"For '{info['question']}':\n{info['summary']}" 
                        for info in all_new_information])
            + "\n\nBased on all this information, provide a comprehensive answer that:\n"
            "1. Addresses the original question\n"
            "2. Incorporates the new context and information gathered\n"
            "3. Addresses the gaps identified in our previous analysis\n"
            "4. Explains relationships and patterns found in the data\n"
            "5. Provides a more complete understanding of the question asked\n"
        )
        
        comprehensive_synthesis = llm.invoke(synthesis_prompt).content
        
        return {
            **state,
            "generation": comprehensive_synthesis,
            "documents": (state["documents"] + 
                        [info["summary"] for info in all_new_information]),
            "subquestions": []  # Clear subquestions as we've processed them all
        }
    except Exception as e:
        print(f"[EXPAND] Error processing queries: {str(e)}")
        return state

def regenerate_with_context(state):
    """Create final comprehensive answer"""
    print("\n[REGENERATE] Starting final synthesis")
    state["current_phase"] = "regenerate_with_context"
    
    synthesis_prompt = f"""
    Original question: {state['question']}
    Initial answer: {state['initial_answer']}
    
    All gathered information:
    {state['documents']}
    
    Initial gaps identified:
    {state.get('analysis', '')}
    
    Create a comprehensive answer that:
    1. States the core finding (what is the top product)
    2. Explains WHY this is the top product using all evidence gathered
    3. Connects different pieces of information into a coherent narrative
    4. Provides business context and causal factors
    5. Addresses the gaps identified in our initial analysis
    
    Focus on creating a clear explanation of both WHAT and WHY.
    """
    
    comprehensive_answer = llm.invoke(synthesis_prompt).content
    print(f"[REGENERATE] Generated comprehensive answer")
    
    return {
        **state,
        "generation": comprehensive_answer,
        "documents": state["documents"] + [comprehensive_answer]
    }

def decide_next(state):
    """Enhanced decision logic with explicit transition logging"""
    print("\n[DECIDE] Evaluating next step:")
    print(f"Current phase: {state.get('current_phase', 'unknown')}")
    print(f"Context iterations: {state.get('context_iterations', 0)}")
    print(f"Scores - Hallucination: {state['hallucination']} | Answer: {state['answer']} | Causal: {state.get('causal_explanation', 'no')}")
    
    decision = None
    
    if state["answer"] == "no" or state["hallucination"] == "yes":
        if state.get("iteration", 0) >= 3:
            decision = "end"
            reason = "Max basic iterations reached with unsatisfactory answer"
        else:
            decision = "regenerate"
            reason = "Answer needs improvement"
    
    elif not state.get("is_multi_hop", False):
        decision = "expand"
        reason = "Starting multi-hop expansion"
    
    elif state.get("context_iterations", 0) < 3:
        if state.get("causal_explanation", "no") == "no":
            decision = "continue_expansion"
            reason = "Need better causal explanation"
        else:
            decision = "synthesize"
            reason = "Good causal explanation achieved, moving to final synthesis"
    
    else:
        decision = "end"
        reason = "Max context iterations reached"
    
    print(f"[DECIDE] Decision: {decision}")
    print(f"[DECIDE] Reason: {reason}")
    return decision

# %%
# Set up the graph
workflow = lg.StateGraph(GraphState)

# Add nodes
workflow.add_node("initial_query", analyze_initial_sql_node)
workflow.add_node("grade", grade)
workflow.add_node("retrieve", retrieve_previous_results)
workflow.add_node("regenerate", regenerate_with_context)
workflow.add_node("retrieve_expansion", retrieve_for_expansion)
workflow.add_node("expand", expand_knowledge)
workflow.add_node("regenerate_with_context", regenerate_with_context)

# Add edges
workflow.add_edge("initial_query", "grade")
workflow.add_edge("regenerate", "grade")
workflow.add_edge("expand", "grade")
workflow.add_edge("regenerate_with_context", "grade")
workflow.add_edge("retrieve_expansion", "expand")  # Critical edge
workflow.add_edge("retrieve", "regenerate")

# Add conditional edges
workflow.add_conditional_edges(
    "grade",
    decide_next,
    {
        "regenerate": "retrieve",
        "expand": "retrieve_expansion",
        "continue_expansion": "expand",
        "synthesize": "regenerate_with_context",
        "end": lg.END
    }
)

workflow.set_entry_point("initial_query")
app = workflow.compile()

# Get Mermaid diagram
mermaid_text = app.get_graph().draw_mermaid()
print(mermaid_text)

# Run the workflow
result = app.invoke({
    "question": "How many items were sold in January 2024?",
    "iteration": 0,
    "context_iterations": 0,
    "is_multi_hop": False,
    "documents": [],
    "subquestions": [],
    "all_subquestions": []
})

# %%
