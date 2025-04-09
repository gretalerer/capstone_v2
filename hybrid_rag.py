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

# Graph state
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

# Vector store
embedding = OpenAIEmbeddings()
documents = [Document(page_content="Initial placeholder document.")]
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# Prompts
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user question to optimize it for retrieval, focusing on causal factors and broader context."),
    ("human", "Original question: {question}\nRewritten query:")
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")

structured_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Does the document help answer the question or support a causal explanation for the answer?"),
    ("human", "Document: {document}\nQuestion: {question}")
])
retrieval_grader = grade_prompt | structured_grader

causal_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate an answer and a causal explanation based only on the provided documents."),
    ("human", "Question: {question}\nContext: {context}\nCausal Insight:")
])
rag_chain = causal_prompt | llm | StrOutputParser()

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="'no' if grounded in facts, 'yes' if hallucinated")
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the generation fully grounded in the facts from the documents?"),
    ("human", "Documents: {documents}\nGeneration: {generation}")
])
hallucination_grader = hallucination_prompt | structured_grader

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="'yes' if it answers and explains the question")
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Does the generation address and explain the question meaningfully?"),
    ("human", "Question: {question}\nGeneration: {generation}")
])
answer_grader = answer_prompt | structured_grader

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# %%
# Nodes

def analyze_initial_sql_node(state):
    sql_query = write_query(state["question"])
    result = execute_query(sql_query)
    summary = summarize_result(sql_query, result, state["question"])
    
    # Add the initial SQL result to the vector store
    initial_doc = Document(page_content=summary)
    vectorstore.add_documents([initial_doc])
    
    # Grade the initial SQL result directly
    hallucination_score = hallucination_grader.invoke({
        "documents": [summary],
        "generation": summary
    }).binary_score
    answer_score = answer_grader.invoke({
        "question": state["question"],
        "generation": summary
    }).binary_score
    
    return {
        **state,
        "initial_answer": summary,
        "documents": [summary],
        "generation": summary,
        "hallucination": hallucination_score,
        "answer": answer_score,
        "came_from_subq": False
    }

def retrieve(state):
    print(f"\n[RETRIEVE] Starting for question: {state['question']}")
    
    # Self-evaluation of retrieval needs
    retrieval_eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate what kind of information is needed to answer this question. Consider both direct facts and contextual information."),
        ("human", "Question: {question}\nCurrent documents: {documents}")
    ])
    retrieval_eval = retrieval_eval_prompt | llm | StrOutputParser()
    retrieval_needs = retrieval_eval.invoke({
        "question": state["question"],
        "documents": state.get("documents", [])
    })
    print(f"[RETRIEVE] Self-evaluation of needs: {retrieval_needs}")
    
    # Rewrite question based on self-evaluation
    rewritten_question = question_rewriter.invoke({"question": state["question"]})
    print(f"[RETRIEVE] Rewritten question: {rewritten_question}")
    
    # Retrieve and grade documents
    docs = retriever.invoke(rewritten_question)
    graded_docs = []
    for doc in docs:
        score = retrieval_grader.invoke({
            "document": doc.page_content,
            "question": state["question"]
        }).binary_score
        print(f"[RETRIEVE] Document: {doc.page_content[:50]}... | Relevant? {score}")
        if score == "yes":
            graded_docs.append(doc.page_content)
    
    # Only mark as sufficient if we have at least one relevant document
    retrieval_sufficiency = {
        "sufficient": len(graded_docs) > 0,
        "needs_more_info": len(graded_docs) == 0
    }
    print(f"[RETRIEVE] Retrieval sufficiency: {retrieval_sufficiency}")
    
    return {
        **state,
        "documents": graded_docs,
        "question": state["question"],
        "iteration": state.get("iteration", 0),
        "retrieval_evaluation": retrieval_sufficiency
    }

def generate(state):
    print(f"\n[GENERATE] Starting with {len(state['documents'])} documents")
    
    # Self-evaluation before generation
    pre_generation_eval = ChatPromptTemplate.from_messages([
        ("system", "Evaluate the documents and plan how to use them to answer the question."),
        ("human", "Question: {question}\nDocuments: {documents}")
    ]) | llm | StrOutputParser()
    
    generation_plan = pre_generation_eval.invoke({
        "question": state["question"],
        "documents": state["documents"]
    })
    print(f"[GENERATE] Generation plan: {generation_plan}")
    
    # Generate with self-evaluation
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate an answer that:
        1. Is grounded in the provided documents
        2. Directly addresses the question
        3. Includes self-evaluation of its own quality
        
        Generation plan: {plan}"""),
        ("human", "Question: {question}\nDocuments: {documents}")
    ])
    
    generation_chain = generation_prompt | llm | StrOutputParser()
    generation = generation_chain.invoke({
        "question": state["question"],
        "documents": state["documents"],
        "plan": generation_plan
    })
    
    # Extract self-evaluation from generation
    self_eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the self-evaluation from the generation."),
        ("human", "Generation: {generation}")
    ]) | llm | StrOutputParser()
    
    self_evaluation = self_eval_prompt.invoke({"generation": generation})
    print(f"[GENERATE] Self-evaluation: {self_evaluation}")
    
    return {
        **state,
        "generation": generation,
        "generation_plan": generation_plan,
        "self_evaluation": self_evaluation
    }

def grade(state):
    print(f"\n[GRADE] Starting grading for question: {state['question']}")
    
    # Grade hallucination
    hallucination_score = hallucination_grader.invoke({
        "documents": state["documents"],
        "generation": state["generation"]
    }).binary_score
    print(f"[GRADE] Hallucination score: {hallucination_score}")
    
    # Grade answer quality - modified to accept basic factual answers
    answer_score = answer_grader.invoke({
        "question": state["question"],
        "generation": state["generation"]
    }).binary_score
    print(f"[GRADE] Answer score: {answer_score}")
    
    # Self-evaluation of answer completeness
    self_eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if the answer is complete and provides sufficient context. Consider that basic factual answers are acceptable as a starting point."),
        ("human", "Question: {question}\nAnswer: {generation}\nDocuments: {documents}")
    ])
    self_eval = self_eval_prompt | llm | StrOutputParser()
    self_evaluation = self_eval.invoke({
        "question": state["question"],
        "generation": state["generation"],
        "documents": state["documents"]
    })
    print(f"[GRADE] Self-evaluation: {self_evaluation}")
    
    return {
        **state,
        "hallucination": hallucination_score,
        "answer": answer_score,
        "self_evaluation": self_evaluation
    }

def decide_next(state):
    # First check if we have a factually correct answer
    if state.get("hallucination") == "no":
        # If we've already generated subquestions and retrieved more info, move to final synthesis
        if state.get("came_from_subq") and len(state.get("all_subquestions", [])) > 0:
            print("[DECIDE] Moving to final synthesis as we have subquestion results")
            return "final_synthesis"
        # Otherwise, generate subquestions to get more specific data
        print("[DECIDE] Moving to subquestions to gather more specific data")
        return "generate_subquestions"
    # If answer is not correct, continue retrieving
    elif state.get("retrieval_evaluation", {}).get("needs_more_info", False):
        print("[DECIDE] Retrieving more information")
        return "retrieve"
    elif state.get("self_evaluation", "").lower().find("incomplete") != -1:
        print("[DECIDE] Self-evaluation indicates incompleteness")
        return "retrieve"
    else:
        print("[DECIDE] Defaulting to retrieve")
        return "retrieve"

def generate_subquestions(state):
    prompt = f"""
    Based on the generation: '{state['generation']}', generate 2 specific, data-driven follow-up questions that can be answered using SQL queries.
    Each question should:
    1. Focus on concrete metrics and numbers
    2. Be answerable with a single SQL query
    3. Not ask for causal explanations or broad factors
    4. Be specific to the data points mentioned in the generation
    
    Example format:
    - What was the total order value for [specific country] in [specific time period]?
    - How many unique customers placed orders in [specific country] during [specific time period]?
    """
    subqs = llm_fast.invoke(prompt).content.split("\n")[:2]
    return {
        **state,
        "subquestions": subqs,
        "all_subquestions": state.get("all_subquestions", []) + subqs,
        "came_from_subq": True
    }

def multi_hop_retrieve(state):
    sub_docs = []
    for subq in state["subquestions"]:
        sql_query = write_query(subq)
        result = execute_query(sql_query)
        summary = summarize_result(sql_query, result, subq)
        sub_docs.append(summary if "error" not in summary.lower() else "No data found.")
    return {
        **state,
        "documents": state["documents"] + sub_docs,
        "came_from_subq": False
    }

def validate_expansion(state):
    # Grade the new information from multi-hop retrieval
    hallucination_score = hallucination_grader.invoke({
        "documents": state["documents"],
        "generation": state["generation"]
    }).binary_score
    answer_score = answer_grader.invoke({
        "question": state["question"],
        "generation": state["generation"]
    }).binary_score
    
    return {
        **state,
        "hallucination": hallucination_score,
        "answer": answer_score
    }

def final_synthesis(state):
    base = state.get("generation", "")
    other_docs = [
        d for d in state["documents"]
        if d.lower() not in base.lower() and "error" not in d.lower()
    ]
    deduped = list(dict.fromkeys(doc.strip() for doc in other_docs))
    context = f"Main Insight:\n{base}\n\nExtra Context:\n- " + "\n- ".join(deduped[:3])
    final = rag_chain.invoke({"context": context, "question": state["question"]})
    return {**state, "generation": final.strip()}

# Graph setup
workflow = lg.StateGraph(GraphState)

# Add nodes
workflow.add_node("initial_query", analyze_initial_sql_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade", grade)
workflow.add_node("generate_subquestions", generate_subquestions)
workflow.add_node("multi_hop_retrieve", multi_hop_retrieve)
workflow.add_node("validate_expansion", validate_expansion)
workflow.add_node("final_synthesis", final_synthesis)

# Add edges
workflow.add_edge("initial_query", "generate_subquestions")  # Go directly to subquestions after initial answer
workflow.add_edge("generate_subquestions", "multi_hop_retrieve")
workflow.add_edge("multi_hop_retrieve", "retrieve")  # Start retrieve-generate-grade loop for subquestions
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_next,
    {
        "retrieve": "retrieve",
        "generate_subquestions": "generate_subquestions",
        "end": lg.END
    }
)
workflow.add_edge("validate_expansion", "final_synthesis")
workflow.add_edge("final_synthesis", lg.END)

# Set entry point
workflow.set_entry_point("initial_query")

# Compile
app = workflow.compile()

# %%
# Run
initial_state = {
    "question": "Which country had the least orders in 2024?",
    "iteration": 0,
    "came_from_subq": False,
    "all_subquestions": []
}
result = app.invoke(initial_state)
print("Final Answer:", result["generation"])
print("Documents Used:", result["documents"])
print("Subquestions (Last Round):", result.get("subquestions", []))
print("All Subquestions:", result.get("all_subquestions", []))

# %%
from IPython.display import Image, display
image = app.get_graph().draw_mermaid_png()
display(Image(image))

# %%
