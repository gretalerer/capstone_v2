# %%
"""
# Hybrid RAG Implementation
This notebook combines Self-RAG, Corrective-RAG (CRAG), and Multi-hop RAG for a robust question-answering system.
"""

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.utilities import SQLDatabase
from google.cloud import bigquery
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
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

# SQLDatabase
db = SQLDatabase.from_uri(f"bigquery://{project_id}")

# LLMs
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define Graph State
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
    reflection: str  # Stores the reflection analysis (ISREL, ISSUP, ISUSE)
    gap_analysis: str  # Stores the analysis of causal gaps in the explanation

# Vectorstore setup
embedding = OpenAIEmbeddings()
documents = [Document(page_content="Initial placeholder document.")]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
doc_splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# Prompts and chains
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user question to optimize it for retrieval, focusing on causal factors and broader context."),
    ("human", "Original question: {question}\nRewritten query:")
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess if the document either provides an answer to the question or supports a causal explanation for the given answer to the question."),
    ("human", "Document: {document}\n Question: {question}\n")
])
retrieval_grader = grade_prompt | structured_llm_grader

causal_prompt = ChatPromptTemplate.from_messages([
    ("system", """Generate a factual answer that enhances the initial response with relevant information. 
    Follow these rules strictly:
    1. Start with the initial answer as your first sentence
    2. Only include information that is explicitly present in the documents
    3. Focus on factual connections between data points
    4. If a gap analysis is provided, use it to identify what additional factual information would be relevant
    5. Do not speculate about causes or reasons
    6. Provide only the essential data points and omit extraneous narrative details.
    7. Do not include flowery language or unnecessary details
    
    Your answer should be direct and factual, connecting relevant data points without speculation."""),
    ("human", """Question: {question}
    Context: {context}
    Gap Analysis: {gap_analysis}
    Initial Answer: {initial_answer}
    
    Generate a factual answer with relevant information:""")
])
rag_chain = causal_prompt | llm | StrOutputParser()

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="'yes' if hallucinated, 'no' if grounded")

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess whether everything on the following generated statement is present on the facts in the documents. Everything in the statement should be traced back to the documents."),
    ("human", "Documents: {documents}\n Generation: {generation}")
])
hallucination_grader = hallucination_prompt | structured_llm_grader

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="'yes' if addresses question, 'no' if not")

class ReflectionTokens(BaseModel):
    isrel: str = Field(description="'yes' if evidence directly pertains to question, 'no' if not")
    issup: str = Field(description="'yes' if claims are grounded in documents, 'no' if not")
    isuse: int = Field(description="Score from 1-5 indicating how well the answer provides causal explanation")

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess if the answer explains the question."),
    ("human", "Question: {question}\nGeneration: {generation}")
])
answer_grader = answer_prompt | structured_llm_grader

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the answer for three aspects:
    1. ISREL: Does the evidence directly pertain to the question?
    2. ISSUP: Is each claim or fact in the answer fully grounded in the retrieved documents?
    3. ISUSE: Does the answer provide a causal explanation? Rate from 1 (poor) to 5 (comprehensive).
    
    Return a structured response with these three fields."""),
    ("human", "Question: {question}\nAnswer: {generation}\nDocuments: {documents}")
])
reflection_grader = reflection_prompt | llm.with_structured_output(ReflectionTokens)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Nodes

def analyze_initial_sql_node(state):
    sql_query = write_query(state["question"])
    result = execute_query(sql_query)
    summary = summarize_result(sql_query, result)
    return {
        **state,
        "initial_answer": summary,
        "documents": [summary],
        "came_from_subq": False
    }

def retrieve(state):
    rewritten_question = question_rewriter.invoke({"question": state["question"]})
    docs = retriever.invoke(rewritten_question)
    docs.append(Document(page_content=state.get("initial_answer", "")))

    graded_docs = []
    for doc in docs:
        score = retrieval_grader.invoke({
            "document": doc.page_content,
            "question": state["question"],
            "initial_answer": state.get("initial_answer", "")
        }).binary_score
        if score == "yes":
            graded_docs.append(doc.page_content)
    return {
        "documents": graded_docs,
        "question": state["question"],
        "iteration": state.get("iteration", 0),
        "initial_answer": state.get("initial_answer", ""),
        "came_from_subq": False
    }

def generate(state):
    context = "\n".join(state["documents"]) if state["documents"] else "No relevant data."
    gap_analysis = state.get("gap_analysis", "No gap analysis available.")
    initial_answer = state.get("initial_answer", "")
    
    generation = rag_chain.invoke({
        "context": context, 
        "question": state["question"],
        "gap_analysis": gap_analysis,
        "initial_answer": initial_answer
    })
    return {
        "generation": generation,
        "documents": state["documents"],
        "question": state["question"],
        "iteration": state["iteration"],
        "came_from_subq": state.get("came_from_subq", False)
    }

def multi_hop_retrieve(state):
    sub_docs = []
    for subq in state["subquestions"]:
        sql_query = write_query(subq)
        results = execute_query(sql_query)
        summary = summarize_result(sql_query, results)
        sub_docs.append(summary if "error" not in summary.lower() else "No data found.")
    return {
        **state,
        "documents": state["documents"] + sub_docs,
        "came_from_subq": False
    }

def analyze_causal_gaps(question, answer, documents):
    gap_prompt = ChatPromptTemplate.from_messages([
        ("system", """Examine the answer and its supporting documents to identify gaps in the causal explanation. 
        Specifically, determine which underlying factors, reasons, or contextual details are missing that would make 
        the causal chain in the answer more comprehensive. Also, look for any contextual fallacies or omissions 
        that might weaken the explanation.
        
        Provide a concise summary in one or two sentences outlining what additional evidence or details are needed 
        to fill these gaps. Make sure not to repeat information already present in the retrieved documents."""),
        ("human", """Analyze the following answer for the question: "{question}"

        Answer: {generation}
        Retrieved Documents: {documents}""")
    ])
    
    gap_chain = gap_prompt | llm_fast | StrOutputParser()
    return gap_chain.invoke({
        "question": question,
        "generation": answer,
        "documents": documents
    })

def decide_to_continue(state):
    reflection = state["reflection"]
    
    # Case A: Evidence is relevant and supported but lacks comprehensive causal explanation
    if reflection.isrel == "yes" and reflection.issup == "yes" and reflection.isuse < 4:
        if state["iteration"] < 5:
            print("\n[ANALYSIS] Answer is factually correct but lacks comprehensive causal explanation")
            return "generate_subquestions"
        return "end"
    
    # Case B: Evidence is not relevant or not supported
    elif reflection.isrel == "no" or reflection.issup == "no":
        if state["iteration"] < 5:
            print("\n[ANALYSIS] Answer lacks proper evidence support or relevance - regenerating")
            return "regenerate"
        return "end"
    
    # If we have good evidence and comprehensive explanation, we're done
    return "end"

def final_synthesis(state):
    print("\n[SYNTHESIS] Synthesizing main and subquestion insights")
    base = state.get("generation", "").strip()
    base_lower = base.lower()
    sub_docs = [doc for doc in state.get("documents", []) if doc.lower() not in base_lower and "error" not in doc.lower()]
    deduped = list(dict.fromkeys([d.strip() for d in sub_docs]))
    context = f"Main Answer:\n{base}\n\nAdditional Causal Factors:\n- " + "\n- ".join(deduped[:3])
    final_answer = rag_chain.invoke({"context": context, "question": state["question"]}).strip()
    return {
        **state,
        "generation": final_answer
    }

def grade(state):
    hallucination_score = hallucination_grader.invoke({
        "documents": state["documents"],
        "generation": state["generation"]
    }).binary_score
    
    answer_score = answer_grader.invoke({
        "question": state["question"],
        "generation": state["generation"]
    }).binary_score
    
    # Generate structured reflection tokens
    reflection = reflection_grader.invoke({
        "question": state["question"],
        "generation": state["generation"],
        "documents": state["documents"]
    })
    
    print("\n[REFLECTION] Analysis:")
    print(f"ISREL: {reflection.isrel}")
    print(f"ISSUP: {reflection.issup}")
    print(f"ISUSE: {reflection.isuse}")
    
    # Generate gap analysis if needed
    gap_analysis = ""
    if reflection.isrel == "yes" and reflection.issup == "yes" and reflection.isuse < 4:
        gap_analysis = analyze_causal_gaps(state["question"], state["generation"], state["documents"])
        print("[ANALYSIS] Gaps identified:", gap_analysis)
    
    return {
        **state,
        "hallucination": hallucination_score,
        "answer": answer_score,
        "iteration": state["iteration"] + 1,
        "reflection": reflection,
        "gap_analysis": gap_analysis
    }

def generate_subquestions(state):
    print(f"\n[SUBQUESTIONS] Generating subquestions for: {state['generation'][:50]}...")
    
    # Include gap analysis in the prompt if available
    gap_context = ""
    if state.get("gap_analysis"):
        gap_context = f"""
        The following gaps were identified in the causal explanation:
        {state['gap_analysis']}
        
        Generate questions that specifically target these gaps to fill in the missing causal factors.
        """
    
    prompt = f"""
    The objective is to find a causal explanation for the answer '{state['generation']}' to the question '{state['question']}'.
    The following data is present in the database: {db.get_table_info()}
    {gap_context}
    Based on the generation, and the already retrieved context, what other internal information is needed to find a causal explanation?
    Formulate 2 follow-up questions that can be answered by the SQL database, looking to find other reasons that can explain the answer.
    - Focus on explaining causal factors or additional details.
    - Do not ask questions that are already answered in the retrieved context.
    - Ensure they are answerable.
    
    Format your response exactly like this:
    1. [First question]
    2. [Second question]
    
    Do not include any other text or explanations in your response.
    """
    new_subquestions = llm_fast.invoke(prompt).content.split("\n")[:2]
    all_subquestions = state.get("all_subquestions", []) + new_subquestions
    print(f"[SUBQUESTIONS] Subquestions: {new_subquestions}")
    return {
        **state,
        "subquestions": new_subquestions,
        "all_subquestions": all_subquestions,
        "came_from_subq": True
    }

# %%
workflow = lg.StateGraph(GraphState)
workflow.add_node("initial_query", analyze_initial_sql_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade", grade)
workflow.add_node("generate_subquestions", generate_subquestions)
workflow.add_node("multi_hop_retrieve", multi_hop_retrieve)

workflow.add_edge("initial_query", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")
workflow.add_conditional_edges("grade", decide_to_continue, {
    "generate_subquestions": "generate_subquestions",
    "multi_hop_retrieve": "multi_hop_retrieve",
    "regenerate": "generate",
    "end": lg.END
})
workflow.add_edge("generate_subquestions", "multi_hop_retrieve")
workflow.add_edge("multi_hop_retrieve", "generate")
workflow.set_entry_point("initial_query")
app = workflow.compile()

# %%
initial_state = {"question": "What is the percentage of orders that were returned in 2024?", "iteration": 0, "came_from_subq": False, "all_subquestions": []}
result = app.invoke(initial_state)
print("Final Answer:", result["generation"])
print("Documents Used:", result["documents"])
print("Subquestions (Last Round):", result.get("subquestions", []))
print("All Subquestions:", result.get("all_subquestions", []))

# %%
# Get the Mermaid code directly
mermaid_code = app.get_graph().draw_mermaid()
print(mermaid_code)
# %%
