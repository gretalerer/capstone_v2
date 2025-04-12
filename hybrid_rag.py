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
import numpy as np
from langchain_core.documents import Document

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
    ("system", """Rewrite the user question to optimize it for retrieval while maintaining its original intent.
    The rewritten query should:
    1. Keep the core question and its main objective
    2. Incorporate specific aspects from the gap analysis that need to be addressed
    3. Focus on retrieving information that fills the identified gaps
    4. Maintain the original context and scope of the question
    
    Return only the rewritten question, without any explanations."""),
    ("human", """Original question: {question}
    Gap Analysis: {gap_analysis}
    Rewritten query:""")
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
    

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def filter_by_similarity(doc_embeddings, query_embedding, docs, similarity_threshold=0.7):
    """
    Filters out documents below a certain cosine similarity threshold.
    Returns (filtered_docs, filtered_embeddings)
    """
    filtered_docs = []
    filtered_embeddings = []
    for doc, emb in zip(docs, doc_embeddings):
        sim = cosine_similarity(emb, query_embedding)
        if sim >= similarity_threshold:
            filtered_docs.append(doc)
            filtered_embeddings.append(emb)
    return filtered_docs, filtered_embeddings

def deduplicate_docs(docs, embeddings, dedup_threshold=0.9):
    """
    Removes near-duplicate documents based on embedding similarity among themselves.
    Returns deduplicated_docs, deduplicated_embeddings
    """
    used = [False] * len(docs)
    deduped_docs = []
    deduped_embs = []
    for i, emb_i in enumerate(embeddings):
        if used[i]:
            continue
        # Keep this doc
        deduped_docs.append(docs[i])
        deduped_embs.append(embeddings[i])
        for j in range(i + 1, len(embeddings)):
            if not used[j]:
                sim = cosine_similarity(emb_i, embeddings[j])
                if sim >= dedup_threshold:
                    used[j] = True
    return deduped_docs, deduped_embs

def retrieve(state):
    """
    Combined retrieval approach:
    1. Rewrite the query using an LLM (question rewriter).
    2. Retrieve from vectorstore (Chroma).
    3. Filter with an LLM-based retrieval grader (yes/no).
    4. Apply a second pass for vector similarity thresholding.
    5. Deduplicate documents that are near-duplicates.
    """
    # 1. Rewrite question
    rewritten_question = question_rewriter.invoke({
        "question": state["question"],
        "gap_analysis": state.get("gap_analysis", "No gap analysis available.")
    })
    rewritten_query = rewritten_question.strip()
    
    # 2. Retrieve from vectorstore (Chroma)
    candidate_docs = vectorstore.similarity_search(rewritten_query, k=10)
    # Convert to raw text docs for consistency
    candidate_texts = [doc.page_content for doc in candidate_docs]
    
    # 3. LLM-based retrieval grader
    graded_docs = []
    for doc_text in candidate_texts:
        score = retrieval_grader.invoke({
            "document": doc_text,
            "question": state["question"],
            "initial_answer": state.get("initial_answer", "")
        }).binary_score
        if score == "yes":
            graded_docs.append(doc_text)
    
    # 4. Vector similarity thresholding
    # Create embeddings for graded docs and the query
    query_emb = embedding.embed_query(rewritten_query)
    doc_embs = [embedding.embed_query(d) for d in graded_docs]
    docs_passed, embs_passed = filter_by_similarity(doc_embs, query_emb, graded_docs, similarity_threshold=0.7)
    
    # 5. Deduplicate near-duplicate documents
    deduped_docs, deduped_embs = deduplicate_docs(docs_passed, embs_passed, dedup_threshold=0.9)
    
    # Combine with existing state's documents (if you want to keep them in context)
    final_documents = list(state["documents"]) + deduped_docs
    
    return {
        "documents": final_documents,
        "question": state["question"],
        "iteration": state.get("iteration", 0),
        "initial_answer": state.get("initial_answer", ""),
        "came_from_subq": False
    }

def generate(state):
    context = "\n".join(state["documents"]) if state["documents"] else "No relevant data."
    gap_analysis = state.get("gap_analysis", "No gap analysis available.")
    initial_answer = state.get("initial_answer", "")
    previous_generation = state.get("generation", "")
    
    # Track iteration to adjust the prompt based on how many times we've generated
    iteration = state.get("iteration", 0)
    
    # Create a more context-aware prompt that builds upon previous answers
    generation_prompt = f"""
    You are a data analyst creating a comprehensive answer to the question: "{state['question']}"
    
    INITIAL ANSWER:
    {initial_answer}
    
    PREVIOUS GENERATION (if any):
    {previous_generation}
    
    NEW INFORMATION FROM SUBQUESTIONS:
    {context}
    
    GAP ANALYSIS:
    {gap_analysis}
    
    ITERATION: {iteration}
    
    Your task is to create an IMPROVED answer that:
    1. Builds upon the previous generation (if any) rather than starting from scratch
    2. Incorporates the new information from subquestions to address the identified gaps
    3. Provides a more comprehensive causal explanation of the observed patterns
    4. Explicitly connects the new information to the original question
    5. Highlights how the new information helps explain the "why" behind the patterns
    
    If this is the first generation (iteration 0), focus on providing a clear, factual answer.
    If this is a subsequent generation (iteration > 0), focus on enhancing the previous answer with new insights.
    
    Structure your answer to show progression:
    - Start with the core answer to the original question
    - Then explain how the new information enhances our understanding
    - Finally, synthesize the insights into a more comprehensive explanation
    
    Your answer should be clear, concise, and directly address the original question while incorporating all relevant new information.
    """
    
    generation = llm.invoke(generation_prompt).content.strip()
    
    return {
        "generation": generation,
        "documents": state["documents"],
        "question": state["question"],
        "iteration": state["iteration"],
        "came_from_subq": state.get("came_from_subq", False)
    }

def multi_hop_retrieve(state):
    print(f"\n[MULTI-HOP] Retrieving for subquestions: {state['subquestions']}")
    sub_docs = []
    for subq in state["subquestions"]:
        sql_query = write_query(subq)
        print(f"[MULTI-HOP] Subquestion SQL: {sql_query}")
        results = execute_query(sql_query)
        print(f"[MULTI-HOP] Subquestion Result: {results[:50]}...")
        summary = summarize_result(sql_query, results)
        if "error" not in summary.lower():
            sub_docs.append(summary)
        else:
            print(f"[MULTI-HOP] SQL failed for {subq}, trying web search")
            web_results = tavily.search(subq, max_results=1)["results"]
            sub_docs.append(f"{web_results[0]['title']}: {web_results[0]['content'][:200]}..." if web_results else "No data found.")
    
    # Add new documents to vectorstore
    if sub_docs:
        new_docs = [Document(page_content=doc) for doc in sub_docs]
        vectorstore.add_documents(new_docs)
        print(f"[MULTI-HOP] Added {len(new_docs)} new documents to vectorstore")
    
    combined_docs = state["documents"] + sub_docs
    print(f"[MULTI-HOP] Total documents after multi-hop: {len(combined_docs)}")
    return {
        "documents": combined_docs,
        "question": state["question"],
        "generation": state["generation"],
        "iteration": state["iteration"]
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
    
    # Get previously asked questions to avoid repetition
    previous_questions = state.get("all_subquestions", [])
    previous_questions_text = "\n".join([f"- {q}" for q in previous_questions]) if previous_questions else "No previous questions have been asked yet."
    
    prompt = f"""
    The objective is to find a causal explanation for the answer '{state['generation']}' to the question '{state['question']}'.
    The following data is present in the database: {db.get_table_info()}
    {gap_context}
    
    PREVIOUSLY ASKED QUESTIONS:
    {previous_questions_text}
    
    Based on the generation, the already retrieved context, and the previously asked questions, formulate 2 NEW follow-up questions that will help EXPLAIN WHY the observed patterns exist.
    
    IMPORTANT GUIDELINES FOR SUBQUESTIONS:
    1. Each question should investigate a potential CAUSAL FACTOR that might explain the observed patterns
    2. Focus on EXPLANATORY VARIABLES that could influence the main metrics
    3. Include specific time periods, locations, or identifiers to make questions precise
    4. Questions should be directly translatable to SQL queries
    5. DO NOT repeat or rephrase questions that have already been asked
    6. Build upon previous questions - if a previous question revealed X, ask about Y that might be related to X
    7. If a previous question didn't yield useful insights, try a different approach or angle
    
    Examples of GOOD causal questions that build on previous findings:
    - If a previous question showed high delivery times in China: "What is the average distance from distribution centers to delivery locations in China compared to Brazil?"
    - If a previous question showed peak hours impact: "How does the number of available delivery personnel during peak hours (9am-5pm) correlate with delivery times?"
    - If a previous question showed order volume impact: "What is the ratio of orders to available delivery vehicles in each country?"
    
    Examples of BAD questions:
    - "What is the average delivery time for orders by country?" (already answered in the initial response)
    - "How many orders were delivered in July 2023?" (doesn't explain causality)
    - "What is the total revenue for each country?" (not related to delivery times)
    - Any question that simply rephrases a previously asked question
    
    Format your response exactly like this:
    1. [First new causal question]
    2. [Second new causal question]
    
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
workflow.add_edge("multi_hop_retrieve", "retrieve")
workflow.set_entry_point("initial_query")
app = workflow.compile()

# %%
initial_state = {"question": "What is the average delivery time for orders by country?", "iteration": 0, "came_from_subq": False, "all_subquestions": []}
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
