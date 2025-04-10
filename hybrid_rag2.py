"""
# Simplified SELF-RAG Implementation
A clean, schematic implementation of a Self-RAG system for Business Intelligence.
"""

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

# Load environment variables
load_dotenv()
client = bigquery.Client()
project_id = client.project
db = SQLDatabase.from_uri(f"bigquery://{project_id}")

print("loaded database")

# LLMs
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("loaded llms")

# Vector store
embedding = OpenAIEmbeddings()
documents = [Document(page_content="Initial placeholder document.")]
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# Define state structure
class GraphState(TypedDict):
    question: str
    initial_answer: str
    documents: List[Document]
    evaluation: Dict[str, Any]
    planned_queries: List[str]
    additional_results: List[Dict[str, Any]]
    final_answer: str
    iteration: int
    current_phase: str  # Track which phase we're in (initial, evaluation, retrieval, generation)
    knowledge_gaps: List[str]  # Track identified knowledge gaps
    causal_insights: List[str]  # Track identified causal insights
    retrieval_history: List[str]  # Track what has been retrieved
    addressed_gaps: List[str]
    remaining_gaps: List[str]

# State manager to handle state updates consistently
def update_state(state: GraphState, updates: Dict[str, Any]) -> GraphState:
    """Update state with new values in a consistent way"""
    new_state = state.copy()
    new_state.update(updates)
    return new_state

# Node functions
def initial_query(state):
    """Get initial answer from SQL"""
    print("\n[INITIAL_QUERY] Starting initial query analysis")
    
    sql_query = write_query(state["question"])
    print(f"[INITIAL_QUERY] Generated SQL: {sql_query}")
        
    result = execute_query(sql_query)
    print("[INITIAL_QUERY] Executed SQL query")
        
    summary = summarize_result(sql_query, result, state["question"])
    print(f"[INITIAL_QUERY] Generated summary: {summary}")
    
    # Add to vector store
    initial_doc = Document(page_content=summary)
    vectorstore.add_documents([initial_doc])
    
    # Create new state with consistent update method
    return update_state(state, {
        "initial_answer": summary,
        "documents": [Document(page_content=summary)],
        "iteration": 0,
        "current_phase": "initial",
        "knowledge_gaps": [],
        "causal_insights": [],
        "retrieval_history": [sql_query]
    })

def evaluate(state):
    """Evaluate the current answer and identify gaps"""
    print("\n[EVALUATE] Evaluating current answer")
    
    # Get current information
    current_answer = state.get("final_answer", state.get("initial_answer", ""))
    documents = state.get("documents", [])
    documents_content = [doc.page_content for doc in documents]
    previous_gaps = state.get("knowledge_gaps", [])
    retrieval_history = state.get("retrieval_history", [])
    
    # Get database schema information
    schema_info = db.get_table_info()
    
    # First prompt: Get analysis of what's missing
    analysis_prompt = f"""You are a BI analyst evaluating what additional information would enhance this answer.

Current Answer:
{current_answer}

SQL Results Used:
{documents_content}

Previously Identified Knowledge Gaps:
{previous_gaps}

Evaluate what additional information would make this answer more insightful. Consider:
1. What context would help explain these numbers?
2. What comparisons would provide more insight?
3. What patterns or trends would be relevant?
4. Which of the previously identified gaps have NOT been addressed yet?

Provide a brief analysis of what information is still missing.
"""
    
    # Second prompt: Generate SQL queries
    query_prompt = f"""Based on the analysis of what information is still missing from this answer:

{current_answer}

Generate 1-3 specific SQL queries that would provide the missing information.
Focus only on information we can get from our database.

Previously Identified Gaps:
{previous_gaps}

Queries Already Executed:
{retrieval_history}

Here is the database schema to help you create valid queries:

{schema_info}

IMPORTANT: 
- Provide ONLY the SQL query text, one per line
- Each query should be a complete, valid SQL statement starting with SELECT
- Do not include any explanations, numbering, or additional text
- Do NOT repeat queries that have already been executed
- Focus on queries that will directly address the remaining knowledge gaps
"""
    
    # Get analysis from LLM
    analysis_response = llm.invoke(analysis_prompt)
    analysis = analysis_response.content.strip()
    
    # Get queries from LLM
    query_response = llm.invoke(query_prompt)
    query_text = query_response.content.strip()
    
    # Extract queries - simple line-by-line approach
    queries = []
    for line in query_text.split('\n'):
        line = line.strip()
        if line.upper().startswith('SELECT'):
            # Check if this query is similar to any previously executed query
            is_duplicate = False
            for prev_query in retrieval_history:
                # Simple similarity check - could be improved
                if line.lower().replace(" ", "") in prev_query.lower().replace(" ", "") or prev_query.lower().replace(" ", "") in line.lower().replace("", ""):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                queries.append(line)
    
    # Extract knowledge gaps from analysis
    # Look for lines that start with bullet points or numbers
    new_gaps = []
    for line in analysis.split('\n'):
        line = line.strip()
        if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line)):
            gap = re.sub(r'^[•\-*\d\.\s]+', '', line).strip()
            if gap and gap not in previous_gaps:
                new_gaps.append(gap)
    
    # If no structured gaps found, try to extract from the analysis
    if not new_gaps:
        # Look for sentences that indicate missing information
        sentences = re.split(r'[.!?]+', analysis)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and ('missing' in sentence.lower() or 'need' in sentence.lower() or 'would help' in sentence.lower()):
                new_gaps.append(sentence)
    
    # Combine previous and new gaps, removing duplicates
    all_gaps = list(set(previous_gaps + new_gaps))
    
    # Update state with consistent update method
    return update_state(state, {
        "evaluation": {
            "analysis": analysis,
            "planned_queries": queries,
            "new_gaps": new_gaps
        },
        "planned_queries": queries,
        "iteration": state.get("iteration", 0) + 1,
        "current_phase": "evaluation",
        "knowledge_gaps": all_gaps
    })

def retrieve(state):
    """Execute SQL queries for the knowledge gaps identified during evaluation"""
    print("\n[RETRIEVE] Executing SQL queries for knowledge gaps")
    
    # Get planned queries from evaluation
    planned_queries = state.get("planned_queries", [])
    if not planned_queries:
        print("No planned queries to execute")
        return state
    
    # Execute each planned query
    additional_results = []
    retrieval_history = state.get("retrieval_history", [])
    
    for query in planned_queries:
        # Skip queries that don't look like SQL
        if not query.upper().startswith('SELECT'):
            print(f"Skipping invalid query: {query}")
            continue
            
        try:
            print(f"Executing query: {query}")
            result = execute_query(query)
            summary = summarize_result(query, result, query)
            additional_results.append({
                "query": query,
                "result": result,
                "summary": summary
            })
            retrieval_history.append(query)
        except Exception as e:
            print(f"Error executing query: {str(e)}")
    
    # Update state with consistent update method
    return update_state(state, {
        "additional_results": additional_results,
        "current_phase": "retrieval",
        "retrieval_history": retrieval_history
    })

def generate(state):
    """Generate a comprehensive answer based on all gathered information"""
    print("\n[GENERATE] Creating comprehensive answer")
    
    # Get all gathered information
    initial_answer = state.get("initial_answer", "")
    documents = state.get("documents", [])
    documents_content = [doc.page_content for doc in documents]
    additional_results = state.get("additional_results", [])
    knowledge_gaps = state.get("knowledge_gaps", [])
    
    # Create a prompt for generating a comprehensive answer
    generation_prompt = f"""You are a business analyst creating a comprehensive analysis of sales data.

Initial Answer:
{initial_answer}

SQL Query Results:
{documents_content}

Additional Information:
{additional_results}

Knowledge Gaps to Address:
{knowledge_gaps}

Create a comprehensive analysis that focuses on EXPLAINING WHY the observed patterns and trends are occurring, not just describing what they are.

Your analysis should:
1. Identify the key causal relationships in the data
2. Explain the "why" behind the numbers and trends
3. Connect different data points to form a coherent narrative
4. Highlight potential business implications of these causal relationships
5. Suggest possible actions based on the causal insights
6. Explicitly address each of the identified knowledge gaps

For each knowledge gap, explain how the new information helps address it.

Guidelines:
- Start with the main causal insight (what's driving the observed patterns)
- Present supporting evidence for your causal explanations
- Use clear language to connect cause and effect
- Keep the tone professional but accessible
- Do not make assumptions without evidence
- Incorporate all relevant information gathered
- Be specific about which data points support which conclusions

Format the output like this:
CAUSAL INSIGHT:
[Main causal relationship in one clear sentence]

EXPLANATION:
• [Supporting evidence for the causal relationship]
• [Additional causal factors]
• [Business implications]

ADDRESSED GAPS:
• [Knowledge gap 1]: [How it was addressed]
• [Knowledge gap 2]: [How it was addressed]
...

REMAINING GAPS:
• [Any knowledge gaps that could not be addressed with current data]

Please provide ONLY the causal analysis, no additional commentary or explanations.
"""

    # Get summary from LLM
    summary_response = llm.invoke(generation_prompt)
    summary = summary_response.content.strip()
    
    # Extract causal insights
    causal_insight_match = re.search(r"CAUSAL INSIGHT:\s*(.+?)(?=EXPLANATION:|$)", summary, re.DOTALL)
    causal_insight = causal_insight_match.group(1).strip() if causal_insight_match else ""
    
    # Extract addressed gaps
    addressed_gaps_match = re.search(r"ADDRESSED GAPS:\s*((?:•.*?\n?)*)", summary, re.DOTALL)
    addressed_gaps_text = addressed_gaps_match.group(1) if addressed_gaps_match else ""
    
    # Parse addressed gaps
    addressed_gaps = []
    if addressed_gaps_text:
        for line in addressed_gaps_match.group(1).split('\n'):
            line = line.strip()
            if line.startswith('•'):
                gap_text = line[1:].strip()
                if ':' in gap_text:
                    gap, explanation = gap_text.split(':', 1)
                    addressed_gaps.append(gap.strip())
    
    # Extract remaining gaps
    remaining_gaps_match = re.search(r"REMAINING GAPS:\s*((?:•.*?\n?)*)", summary, re.DOTALL)
    remaining_gaps_text = remaining_gaps_match.group(1) if remaining_gaps_match else ""
    
    # Parse remaining gaps
    remaining_gaps = []
    if remaining_gaps_text:
        for line in remaining_gaps_match.group(1).split('\n'):
            line = line.strip()
            if line.startswith('•'):
                gap = line[1:].strip()
                remaining_gaps.append(gap)
    
    # Update state with consistent update method
    return update_state(state, {
        "final_answer": summary,
        "current_phase": "generation",
        "causal_insights": [causal_insight] if causal_insight else [],
        "addressed_gaps": addressed_gaps,
        "remaining_gaps": remaining_gaps
    })

def decide_next(state):
    """Decide whether to continue or end the process"""
    print("\n[DECIDE] Evaluating next steps")
    
    # Get current state
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
    print(f"Total knowledge gaps: {len(knowledge_gaps)}")
    
    # If we've reached max iterations, end
    if iteration >= 3:
        print("Decision: End - Reached maximum iterations")
        return "end"
    
    # If we have no planned queries, end
    if not planned_queries:
        print("Decision: End - No more queries to execute")
        return "end"
    
    # Check if we have valid SQL queries
    valid_queries = [q for q in planned_queries if q.upper().startswith('SELECT')]
    if not valid_queries:
        print("Decision: End - No valid SQL queries to execute")
        return "end"
    
    # If we have addressed all knowledge gaps, end
    if addressed_gaps and not remaining_gaps:
        print("Decision: End - All knowledge gaps have been addressed")
        return "end"
    
    # If we have causal insights and no remaining gaps, end
    if causal_insights and not remaining_gaps:
        print("Decision: End - Sufficient causal insights achieved with no remaining gaps")
        return "end"
    
    # If we have no progress in addressing gaps between iterations, end
    if iteration > 1 and len(addressed_gaps) == 0:
        print("Decision: End - No progress in addressing gaps")
        return "end"
    
    # Otherwise, continue with retrieval
    print("Decision: Continue - More information needed")
    return "retrieve"

# Set up the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("initial_query", initial_query)
workflow.add_node("evaluate", evaluate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Add edges
workflow.add_edge("initial_query", "evaluate")
workflow.add_edge("evaluate", "retrieve")
workflow.add_edge("retrieve", "generate")

# Add conditional edges
workflow.add_conditional_edges(
    "generate",
    decide_next,
    {
        "retrieve": "evaluate",
        "end": END
    }
)

# Set entry point
workflow.set_entry_point("initial_query")

# Compile the graph
app = workflow.compile()

# Get Mermaid diagram
mermaid_text = app.get_graph().draw_mermaid()
print(mermaid_text)

# Example usage
def run_workflow(question):
    """Run the workflow with a given question"""
    result = app.invoke({
        "question": question,
        "iteration": 0
    })
    return result

# Example
if __name__ == "__main__":
    result = run_workflow("How many items were sold in January 2024?")
    print("\nFinal Answer:")
    print(result["final_answer"])
