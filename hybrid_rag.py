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
import json
import os.path
from eval_RAGBit.test_functions import save_gap_analysis_history, track_retrieval_metrics, gap_analysis_history

# %%
load_dotenv()
client = bigquery.Client()
project_id = client.project

# SQLDatabase
db = SQLDatabase.from_uri(f"bigquery://{project_id}")

# LLMs
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Global variable to track gap analyses and their corresponding subquestions
gap_analysis_history = {}

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

class ReflectionTokens(BaseModel):
    isrel: str = Field(description="'yes' if evidence directly pertains to question, 'no' if not")
    issup: str = Field(description="'yes' if claims are grounded in documents, 'no' if not")
    isuse: int = Field(description="Score from 1-5 indicating how well the answer provides causal explanation: 1=no causal information, 2=basic inferences only, 3=some retrieved data with inferences, 4=comprehensive retrieved data with good inferences, 5=complete causal explanation with multiple data sources")

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the answer for three aspects:
    1. ISREL: Does the evidence directly pertain to the question?
    2. ISSUP: Is each claim or fact in the answer fully grounded in the retrieved documents?
    3. ISUSE: Does the answer provide a causal explanation? Rate from 1-5 using these specific criteria:
       - Score 1: No causal information provided, just descriptive facts
       - Score 2: Basic inferences from internal knowledge only, no retrieved data used for explanation
       - Score 3: Some retrieved data used with basic inferences, but explanation is incomplete
       - Score 4: Comprehensive retrieved data with good inferences, but could use more specific examples
       - Score 5: Complete causal explanation with multiple data sources, specific examples, and clear connections
    
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
        "generation": summary,
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
    
    # 2. Retrieve from vectorstore (Chroma) - INCREASED from 10 to 15 documents
    candidate_docs = vectorstore.similarity_search(rewritten_query, k=15)
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
    
    # 4. Vector similarity thresholding - LOWERED threshold from 0.7 to 0.6 to include more documents
    # Create embeddings for graded docs and the query
    query_emb = embedding.embed_query(rewritten_query)
    doc_embs = [embedding.embed_query(d) for d in graded_docs]
    docs_passed, embs_passed = filter_by_similarity(doc_embs, query_emb, graded_docs, similarity_threshold=0.6)
    
    # 5. Deduplicate near-duplicate documents - INCREASED threshold from 0.9 to 0.95 to be more lenient
    deduped_docs, deduped_embs = deduplicate_docs(docs_passed, embs_passed, dedup_threshold=0.95)
    
    # Track retrieval metrics for noise filtering analysis
    track_retrieval_metrics(
        initial_docs=candidate_texts,
        filtered_docs=docs_passed,
        deduped_docs=deduped_docs,
        query=state["question"],
        similarity_threshold=0.6,
        dedup_threshold=0.95
    )
    
    # Combine with existing state's documents
    final_documents = list(state["documents"]) + deduped_docs
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in final_documents:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
    
    print(f"[RETRIEVAL] Retrieved {len(unique_docs)} unique documents")
    
    return {
        "documents": unique_docs,
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
    3. Provides a more comprehensive explanation of the observed patterns
    4. Explicitly connects the new information to the original question
    5. Highlights how the new information helps explain the patterns
    
    BALANCED APPROACH:
    1. Ground your answer primarily in the factual data from the database
    2. Supplement with reasonable inferences and insights based on your knowledge
    3. Clearly distinguish between facts from the data and your inferences
    4. When making inferences, explain your reasoning and acknowledge uncertainty
    5. Avoid wild speculation or claims that contradict the data
    
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
    
    # Process each subquestion
    for subq in state["subquestions"]:
        print(f"[MULTI-HOP] Processing subquestion: {subq}")
        
        # Try SQL query first
        try:
            sql_query = write_query(subq)
            print(f"[MULTI-HOP] Subquestion SQL: {sql_query}")
            results = execute_query(sql_query)
            print(f"[MULTI-HOP] Subquestion Result: {results[:50]}...")
            summary = summarize_result(sql_query, results)
            if "error" not in summary.lower():
                sub_docs.append(summary)
                print(f"[MULTI-HOP] Successfully retrieved SQL data for: {subq}")
            else:
                print(f"[MULTI-HOP] SQL failed for {subq}, trying web search")
                # Try web search if SQL fails
                web_response = tavily.search(subq, max_results=2, search_depth="advanced", include_answer="advanced")
                if web_response.get("answer"):
                    sub_docs.append(f"Web Search Answer: {web_response['answer']}")
        except Exception as e:
            print(f"[MULTI-HOP] Error processing SQL for {subq}: {str(e)}")
            # Try web search if SQL fails
            try:
                web_response = tavily.search(subq, max_results=2, search_depth="advanced", include_answer="advanced")
                if web_response.get("answer"):
                    sub_docs.append(f"Web Search Answer: {web_response['answer']}")
            except Exception as web_error:
                print(f"[MULTI-HOP] Error with web search for {subq}: {str(web_error)}")
    
    # Add new documents to vectorstore
    if sub_docs:
        new_docs = [Document(page_content=doc) for doc in sub_docs]
        vectorstore.add_documents(new_docs)
        print(f"[MULTI-HOP] Added {len(new_docs)} new documents to vectorstore")
    
    # Combine with existing documents
    combined_docs = state["documents"] + sub_docs
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for doc in combined_docs:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
    
    print(f"[MULTI-HOP] Total unique documents after multi-hop: {len(unique_docs)}")
    return {
        "documents": unique_docs,
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
        
        Focus on these specific areas:
        1. COMPARATIVE ANALYSIS: Are there direct comparisons between countries with very different delivery times?
        2. INFRASTRUCTURE FACTORS: Are there details about logistics infrastructure, transportation networks, or delivery systems?
        3. REGULATORY ENVIRONMENT: Are there mentions of regulations, customs processes, or legal frameworks that affect delivery?
        4. CULTURAL FACTORS: Are there considerations of cultural practices, consumer behaviors, or local preferences?
        5. TEMPORAL ASPECTS: Is there information about how delivery times have changed over time?
        6. GEOGRAPHICAL CONSIDERATIONS: Are there details about distances, terrain, or population density?
        7. ECONOMIC FACTORS: Are there mentions of economic conditions, labor costs, or market dynamics?
        8. TECHNOLOGICAL ADOPTION: Is there information about technology use, automation, or digital infrastructure?
        
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
    # Use the should_continue flag from the grade function
    if state.get("should_continue", False):
        print("\n[ANALYSIS] Answer needs improvement - continuing exploration")
        return "generate_subquestions"
    
    # If we shouldn't continue, we're done
    print("\n[ANALYSIS] Answer is comprehensive - ending")
    return "end"

def final_synthesis(state):
    print("\n[SYNTHESIS] Synthesizing main and subquestion insights")
    base = state.get("generation", "").strip()
    base_lower = base.lower()
    sub_docs = [doc for doc in state.get("documents", []) if doc.lower() not in base_lower and "error" not in doc.lower()]
    deduped = list(dict.fromkeys([d.strip() for d in sub_docs]))
    
    # Create a more structured prompt for the final synthesis
    synthesis_prompt = f"""
    You are a data analyst creating a comprehensive answer to the question: "{state['question']}"
    
    INITIAL ANSWER:
    {base}
    
    ADDITIONAL INSIGHTS FROM SUBQUESTIONS:
    {deduped}
    
    Your task is to create a FINAL, COMPREHENSIVE answer that:
    1. Starts with a clear, concise summary of the key findings (the "Core Answer")
    2. Follows with an "Enhancement" section that incorporates new information from subquestions
    3. Concludes with a "Comprehensive Explanation" that provides a detailed causal analysis
    
    STRUCTURE YOUR ANSWER AS FOLLOWS:
    
    **Core Answer:**
    [Provide a clear, factual summary of the main findings, focusing on the direct answer to the question]
    
    **Enhancement with New Information:**
    [Explain how the new information from subquestions enhances our understanding of the initial answer]
    
    **Comprehensive Explanation:**
    [Provide a detailed causal explanation that:
     - Identifies multiple factors that contribute to the observed patterns
     - Explains how these factors interact with each other
     - Compares and contrasts different countries or scenarios
     - Draws on specific data points from both the initial answer and subquestions
     - Acknowledges any limitations or uncertainties in the explanation
     - Concludes with actionable insights or implications]
    
    Your answer should be well-structured, comprehensive, and provide a clear causal explanation for the observed patterns.
    """
    
    final_answer = llm.invoke(synthesis_prompt).content.strip()
    
    return {
        **state,
        "generation": final_answer
    }

def grade(state):
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
    
    # Count how many documents were retrieved (excluding the initial answer)
    retrieved_docs_count = len(state["documents"]) - 1  # Subtract 1 for the initial answer
    
    # Calculate a score based on both the reflection and the amount of retrieved information
    # We want to encourage more retrieval if we haven't gathered enough information yet
    
    # Base score on reflection - More stringent scoring
    base_score = 0
    if reflection.isrel == "yes":
        base_score += 0.25  # Reduced from 0.3
    if reflection.issup == "yes":
        base_score += 0.25  # Reduced from 0.3
    
    # Normalize ISUSE to 0-1 range with stricter criteria
    # A score of 4 or less should result in a low base score to encourage more retrieval
    isuse_score = reflection.isuse / 5
    base_score += isuse_score * 0.5  # ISUSE contributes 50% to base score (increased from 0.4)
    
    # Adjust score based on retrieved information
    # We want at least 5 retrieved documents for a good score (increased from 3)
    retrieval_factor = min(retrieved_docs_count / 5, 1.0)
    
    # Final score is a weighted average of base score and retrieval factor
    # This encourages the system to continue retrieving until it has enough information
    final_score = (base_score * 0.7) + (retrieval_factor * 0.3)  # Increased weight on base score
    
    # Determine if we should continue based on the final score and ISUSE score
    # We want a score of at least 0.85 to consider the answer comprehensive (increased from 0.8)
    # Also, if ISUSE is 4 or less, we should continue regardless of the final score
    should_continue = (final_score < 0.85 or reflection.isuse <= 4) and state["iteration"] < 5
    
    print(f"[GRADING] Base Score: {base_score:.2f}, Retrieval Factor: {retrieval_factor:.2f}")
    print(f"[GRADING] Final Score: {final_score:.2f}, Should Continue: {should_continue}")
    print(f"[GRADING] ISUSE Score: {reflection.isuse}/5, Needs Improvement: {reflection.isuse <= 4}")
    
    # Generate gap analysis if needed
    gap_analysis = ""
    if should_continue:
        gap_analysis = analyze_causal_gaps(state["question"], state["generation"], state["documents"])
        print("[ANALYSIS] Gaps identified:", gap_analysis)
    
    return {
        **state,
        "iteration": state["iteration"] + 1,
        "reflection": reflection,
        "gap_analysis": gap_analysis,
        "final_score": final_score,
        "should_continue": should_continue
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
    
    Based on the generation, the already retrieved context, and the previously asked questions, formulate 3 NEW follow-up questions that will help EXPLAIN WHY the observed patterns exist.
    
    IMPORTANT GUIDELINES FOR SUBQUESTIONS:
    1. Each question should investigate a potential CAUSAL FACTOR that might explain the observed patterns
    2. Focus on EXPLANATORY VARIABLES that could influence the main metrics
    3. Include specific time periods, locations, or identifiers to make questions precise
    4. Questions should be directly translatable to SQL queries
    5. DO NOT repeat or rephrase questions that have already been asked
    6. Build upon previous questions - if a previous question revealed X, ask about Y that might be related to X
    7. If a previous question didn't yield useful insights, try a different approach or angle
    8. Consider COMPARATIVE questions that directly compare countries with very different delivery times
    9. Include questions about INFRASTRUCTURE, REGULATIONS, or CULTURAL FACTORS that might affect delivery times
    10. Consider TEMPORAL aspects - how delivery times have changed over time in different countries
    
    Examples of GOOD causal questions that build on previous findings:
    - If a previous question showed high delivery times in China: "What is the average distance from distribution centers to delivery locations in China compared to Brazil?"
    - If a previous question showed peak hours impact: "How does the number of available delivery personnel during peak hours (9am-5pm) correlate with delivery times?"
    - If a previous question showed order volume impact: "What is the ratio of orders to available delivery vehicles in each country?"
    - "How do delivery times in Spain compare to Colombia during peak shopping seasons?"
    - "What is the correlation between a country's logistics infrastructure score and its average delivery time?"
    - "How have delivery times changed in the past 5 years in countries with the shortest and longest delivery times?"
    
    Examples of BAD questions:
    - "What is the average delivery time for orders by country?" (already answered in the initial response)
    - "How many orders were delivered in July 2023?" (doesn't explain causality)
    - "What is the total revenue for each country?" (not related to delivery times)
    - Any question that simply rephrases a previously asked question
    
    Format your response exactly like this:
    1. [First new causal question]
    2. [Second new causal question]
    3. [Third new causal question]
    
    Do not include any other text or explanations in your response.
    """
    new_subquestions = llm_fast.invoke(prompt).content.split("\n")[:3]
    all_subquestions = state.get("all_subquestions", []) + new_subquestions
    
    # Update the global gap analysis history
    if state.get("gap_analysis"):
        # Use the gap analysis as a key in the dictionary
        gap_key = state["gap_analysis"]
        if gap_key not in gap_analysis_history:
            gap_analysis_history[gap_key] = []
        
        # Add the new subquestions to the list for this gap analysis
        gap_analysis_history[gap_key].extend(new_subquestions)
        
        # Save the updated history to the JSON file
        save_gap_analysis_history()
    
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

workflow.add_edge("initial_query", "grade")
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
initial_state = {"question": "Which product categories generate the highest number of orders overall?", "iteration": 0, "came_from_subq": False, "all_subquestions": []}
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

