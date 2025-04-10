import os
import re
from typing import Dict, List, Tuple, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

from google.cloud import bigquery
from dotenv import load_dotenv
import pandas as pd
from tools.nl2sql_tools import write_query, execute_query, summarize_result

###############################################################################
# LOAD ENVIRONMENT & CONNECT TO DATABASE
###############################################################################

load_dotenv()
client = bigquery.Client()
project_id = client.project
db = SQLDatabase.from_uri(f"bigquery://{project_id}")

# Create LLM instances
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create embedding model
embedding = OpenAIEmbeddings()

###############################################################################
# INITIALIZE VECTORSTORE
# We’ll start with a placeholder doc, then store new docs as we go along.
###############################################################################

documents = [Document(page_content="Initial placeholder document.",
                      metadata={"source": "initial"})]
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

###############################################################################
# TYPEDEFS / STATE
###############################################################################

class GraphState(TypedDict):
    question: str
    initial_answer: str
    documents: List[Document]                  # Documents used so far
    evaluation: Dict[str, Any]
    planned_queries: List[str]
    additional_results: List[Dict[str, Any]]
    final_answer: str
    iteration: int
    current_phase: str
    knowledge_gaps: List[str]
    causal_insights: List[str]
    retrieval_history: List[str]
    addressed_gaps: List[str]
    remaining_gaps: List[str]

###############################################################################
# STATE UTILS
###############################################################################

def update_state(state: GraphState, updates: Dict[str, Any]) -> GraphState:
    """Return a new state dict with the given updates applied."""
    new_state = state.copy()
    new_state.update(updates)
    return new_state

###############################################################################
# GRAPH NODES
###############################################################################

def initial_query(state: GraphState) -> GraphState:
    """
    1) Generate an initial SQL query from the user question,
    2) Execute and summarize the result,
    3) Store the doc and update the vector store.
    """
    print("\n[INITIAL_QUERY] Starting initial query analysis")
    sql_query = write_query(state["question"])
    print(f"[INITIAL_QUERY] Generated SQL:\n{sql_query}\n")
    
    try:
        result = execute_query(sql_query)
        print("[INITIAL_QUERY] Query executed successfully.\n")
    except Exception as e:
        print(f"[INITIAL_QUERY] Error executing initial query: {str(e)}\n")
        return update_state(state, {
            "initial_answer": f"Error executing query: {str(e)}",
            "iteration": 0,
            "current_phase": "initial"
        })
    
    summary = summarize_result(sql_query, result, state["question"])
    print(f"[INITIAL_QUERY] Summary:\n{summary}\n")
    
    # Create a doc for the summary. We add metadata to track where it came from.
    initial_doc = Document(
        page_content=summary,
        metadata={
            "source": "initial_query",
            "sql_query": sql_query
        }
    )
    
    # Add to vector store
    vectorstore.add_documents([initial_doc])
    
    # Update the state's documents
    return update_state(state, {
        "initial_answer": summary,
        "documents": [initial_doc],
        "iteration": 0,
        "current_phase": "initial",
        "knowledge_gaps": [],
        "causal_insights": [],
        "retrieval_history": [sql_query]
    })

def evaluate(state: GraphState) -> GraphState:
    """
    1) Evaluate the current answer and identify what additional information would help.
    2) Generate new SQL queries that might address these gaps.
    3) Parse out the knowledge gaps from the LLM’s analysis.
    """
    print("\n[EVALUATE] Evaluating current answer.")
    
    current_answer = state.get("final_answer", state.get("initial_answer", ""))
    docs = state.get("documents", [])
    doc_texts = [d.page_content for d in docs]
    previous_gaps = state.get("knowledge_gaps", [])
    retrieval_history = state.get("retrieval_history", [])
    
    # Fetch database schema info to help the LLM craft correct queries.
    schema_info = db.get_table_info()
    
    # 1) Ask the LLM to analyze missing pieces
    analysis_prompt = f"""
You are a BI analyst evaluating what additional information would enhance this answer.

CURRENT ANSWER:
{current_answer}

SQL RESULTS USED:
{doc_texts}

PREVIOUSLY IDENTIFIED KNOWLEDGE GAPS:
{previous_gaps}

Evaluate what additional information would make this answer more insightful. Consider:
1. What context would help explain these numbers?
2. What comparisons would provide more insight?
3. What patterns or trends would be relevant?
4. Which of the previously identified gaps have NOT been addressed?

Please provide a concise analysis of what is still missing.
"""
    analysis_response = llm.invoke(analysis_prompt).content.strip()
    
    # 2) Ask the LLM to generate new SQL queries to address missing pieces
    query_prompt = f"""
Based on the above analysis of missing information in the answer:

CURRENT ANSWER:
{current_answer}

Generate 1-3 specific SQL queries that would provide the missing information.
Focus only on information we can get from the database.

PREVIOUS GAPS: {previous_gaps}

QUERIES ALREADY EXECUTED:
{retrieval_history}

DB SCHEMA:
{schema_info}

IMPORTANT:
- Provide ONLY the SQL query text, one per line
- Each query should be a complete, valid SQL statement starting with SELECT
- Do not include any explanations, numbering, or extra text
- Do NOT repeat queries that have already been executed
- Focus on queries that will directly address the remaining knowledge gaps
"""
    query_response = llm.invoke(query_prompt).content.strip()
    
    # Extract queries line-by-line
    queries = []
    for line in query_response.split('\n'):
        candidate = line.strip()
        if candidate.upper().startswith('SELECT'):
            # Check if this query is a duplicate of a prior retrieval
            candidate_no_spaces = candidate.lower().replace(" ", "")
            is_duplicate = False
            for prev in retrieval_history:
                prev_no_spaces = prev.lower().replace(" ", "")
                if candidate_no_spaces in prev_no_spaces or prev_no_spaces in candidate_no_spaces:
                    is_duplicate = True
                    break
            if not is_duplicate:
                queries.append(candidate)
    
    # Extract knowledge gaps from the analysis.  We'll look for lines that appear
    # to describe missing info (bullets, enumerations, or contain "missing"/"need").
    new_gaps = []
    for line in analysis_response.split('\n'):
        line = line.strip()
        if (line.startswith('•') or line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line)):
            gap = re.sub(r'^[•\-\*\d\.\s]+', '', line).strip()
            if gap and gap not in previous_gaps:
                new_gaps.append(gap)
        # fallback: see if the line references “missing,” “need,” or “would help”
        elif any(keyword in line.lower() for keyword in ["missing", "need", "would help"]):
            if line not in previous_gaps:
                new_gaps.append(line)
    
    # Combine old & new gaps
    all_gaps = list(set(previous_gaps + new_gaps))
    
    return update_state(state, {
        "evaluation": {
            "analysis": analysis_response,
            "planned_queries": queries,
            "new_gaps": new_gaps
        },
        "planned_queries": queries,
        "iteration": state.get("iteration", 0) + 1,
        "current_phase": "evaluation",
        "knowledge_gaps": all_gaps
    })

def retrieve(state: GraphState) -> GraphState:
    """
    Execute the planned SQL queries to address the identified knowledge gaps.
    Summarize the results, store them, and update state.
    """
    print("\n[RETRIEVE] Executing SQL queries for knowledge gaps.")
    
    planned_queries = state.get("planned_queries", [])
    retrieval_history = state.get("retrieval_history", [])
    additional_results = state.get("additional_results", [])
    
    if not planned_queries:
        print("[RETRIEVE] No planned queries to execute.\n")
        return state
    
    for query in planned_queries:
        if not query.upper().startswith('SELECT'):
            print(f"[RETRIEVE] Skipping invalid query: {query}\n")
            continue
        print(f"[RETRIEVE] Executing:\n{query}\n")
        try:
            result = execute_query(query)
            summary = summarize_result(query, result, query)
            # Record the result in a doc
            doc = Document(
                page_content=summary,
                metadata={"source": "retrieve_step", "sql_query": query}
            )
            vectorstore.add_documents([doc])
            
            additional_results.append({
                "query": query,
                "result": result,
                "summary": summary
            })
            retrieval_history.append(query)
        except Exception as e:
            print(f"[RETRIEVE] Error executing query: {str(e)}\n")
    
    return update_state(state, {
        "additional_results": additional_results,
        "retrieval_history": retrieval_history,
        "current_phase": "retrieval"
    })

def generate(state: GraphState) -> GraphState:
    """
    Generate a comprehensive answer that explains the WHY behind the data,
    highlighting causal insights and addressing knowledge gaps.
    """
    print("\n[GENERATE] Creating comprehensive answer.")
    
    initial_answer = state.get("initial_answer", "")
    docs = state.get("documents", [])
    docs_text = [doc.page_content for doc in docs]
    
    # Also gather newly retrieved data
    additional_results = state.get("additional_results", [])
    
    knowledge_gaps = state.get("knowledge_gaps", [])
    
    # We'll pass all the doc texts plus the new results
    # (We can just pass the textual summaries for brevity.)
    extra_summaries = []
    for ar in additional_results:
        extra_summaries.append(ar["summary"])
    
    # Prompt the LLM to produce a causal explanation
    generation_prompt = f"""
You are a business analyst creating a comprehensive analysis of sales (or similar) data.

INITIAL ANSWER:
{initial_answer}

ORIGINAL SQL RESULTS:
{docs_text}

ADDITIONAL RESULTS:
{extra_summaries}

KNOWLEDGE GAPS TO ADDRESS:
{knowledge_gaps}

INSTRUCTIONS:
1. Identify the key causal relationships driving the observed patterns.
2. Explain the 'why' behind the numbers and trends (avoid just repeating the data).
3. Connect different data points into a coherent narrative.
4. Highlight potential business implications and next steps.
5. Explicitly address each identified knowledge gap, explaining how new info helps.
6. Keep your output structured under the following sections:

CAUSAL INSIGHT:
[Main causal relationship in one clear sentence]

EXPLANATION:
• [Evidence or data supporting your claims]
• [Possible root causes or influences]
• [Potential business implications]

ADDRESSED GAPS:
• [List each knowledge gap and how the data addresses it]

REMAINING GAPS:
• [List any gaps that still remain unanswered or need more data]

NO extra commentary, just the above structure.
"""
    
    summary_response = llm.invoke(generation_prompt)
    summary = summary_response.content.strip()
    
    # Attempt to parse out some fields from the summary
    # (Optional – you could keep it simpler if you want.)
    causal_insight_match = re.search(
        r"CAUSAL INSIGHT:\s*(.+?)(?=EXPLANATION:|$)",
        summary, flags=re.DOTALL
    )
    causal_insight = causal_insight_match.group(1).strip() if causal_insight_match else ""
    
    addressed_gaps_match = re.search(
        r"ADDRESSED GAPS:\s*((?:•.*?\n?)+)",
        summary, flags=re.DOTALL
    )
    addressed_gaps_text = addressed_gaps_match.group(1) if addressed_gaps_match else ""
    
    # Parse addressed gap lines
    addressed_gaps = []
    for line in addressed_gaps_text.split('\n'):
        line = line.strip()
        if line.startswith('•'):
            text_after_bullet = line[1:].strip()
            if ":" in text_after_bullet:
                gap_title, explanation = text_after_bullet.split(':', 1)
                addressed_gaps.append(gap_title.strip())
    
    remaining_gaps_match = re.search(
        r"REMAINING GAPS:\s*((?:•.*?\n?)+)",
        summary, flags=re.DOTALL
    )
    remaining_gaps_text = remaining_gaps_match.group(1) if remaining_gaps_match else ""
    
    remaining_gaps = []
    for line in remaining_gaps_text.split('\n'):
        line = line.strip()
        if line.startswith('•'):
            gap = line[1:].strip()
            remaining_gaps.append(gap)
    
    return update_state(state, {
        "final_answer": summary,
        "current_phase": "generation",
        "causal_insights": [causal_insight] if causal_insight else [],
        "addressed_gaps": addressed_gaps,
        "remaining_gaps": remaining_gaps
    })

def decide_next(state: GraphState) -> str:
    """
    Decide whether to continue with another round (go to 'evaluate') or end.
    This logic can be tweaked as you see fit.
    """
    print("\n[DECIDE] Evaluating next steps.")
    
    iteration = state.get("iteration", 0)
    planned_queries = state.get("planned_queries", [])
    current_phase = state.get("current_phase", "")
    causal_insights = state.get("causal_insights", [])
    addressed_gaps = state.get("addressed_gaps", [])
    remaining_gaps = state.get("remaining_gaps", [])
    knowledge_gaps = state.get("knowledge_gaps", [])
    
    print(f"Current iteration: {iteration}")
    print(f"Current phase: {current_phase}")
    print(f"Planned queries: {len(planned_queries)}")
    print(f"Causal insights: {len(causal_insights)}")
    print(f"Addressed gaps: {len(addressed_gaps)}")
    print(f"Remaining gaps: {len(remaining_gaps)}")
    print(f"Total knowledge gaps: {len(knowledge_gaps)}\n")
    
    if iteration >= 3:
        print("[DECIDE] Stopping - maximum iterations reached.\n")
        return "end"
    
    # If we have no planned queries, see if there are still unaddressed gaps
    if not planned_queries:
        # If we still have gaps, we could do more, but let's end for now:
        print("[DECIDE] No new queries - stopping.\n")
        return "end"
    
    # If all known gaps are addressed and there's a causal insight, we can stop
    if (not remaining_gaps) and causal_insights:
        print("[DECIDE] All knowledge gaps addressed and we have a causal insight - stopping.\n")
        return "end"
    
    # If no valid new queries or no progress, end
    valid_queries = [q for q in planned_queries if q.upper().startswith('SELECT')]
    if not valid_queries:
        print("[DECIDE] No valid queries to run - stopping.\n")
        return "end"
    
    # Otherwise, continue another iteration
    print("[DECIDE] Continuing for deeper analysis.\n")
    return "retrieve"

###############################################################################
# BUILD THE GRAPH
###############################################################################

workflow = StateGraph(GraphState)

workflow.add_node("initial_query", initial_query)
workflow.add_node("evaluate", evaluate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge("initial_query", "evaluate")
workflow.add_edge("evaluate", "retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_conditional_edges(
    "generate",
    decide_next,
    {
        "retrieve": "evaluate",
        "end": END
    }
)

workflow.set_entry_point("initial_query")
app = workflow.compile()

###############################################################################
# MERMAID DIAGRAM (OPTIONAL)
###############################################################################

mermaid_text = app.get_graph().draw_mermaid()
print("Mermaid Workflow Diagram:\n")
print(mermaid_text)

###############################################################################
# EXAMPLE USAGE
###############################################################################

def run_workflow(question: str) -> GraphState:
    """Run the entire multi-hop workflow for the given question."""
    print("\n=== RUN WORKFLOW ===\n")
    result = app.invoke({"question": question, "iteration": 0})
    return result

if __name__ == "__main__":
    example_question = "How many items were sold in January 2024?"
    final_state = run_workflow(example_question)
    print("\n=== FINAL ANSWER ===\n")
    print(final_state["final_answer"])
