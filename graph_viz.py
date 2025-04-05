"""
Visualization of the Hybrid RAG Graph Structure
"""
from typing import List, TypedDict
import langgraph.graph as lg

# Define minimal state structure for graph visualization
class GraphState(TypedDict):
    question: str
    initial_answer: str
    generation: str
    documents: List[str]
    subquestions: List[str]
    iteration: int

# Create empty placeholder functions
def noop(state): return state
def always_continue(state): return "multi_hop_retrieve" if state.get("iteration", 0) < 3 else "end"

# Build Graph Structure
workflow = lg.StateGraph(GraphState)

# Add nodes (using noop function as placeholder)
workflow.add_node("initial_query", noop)
workflow.add_node("retrieve", noop)
workflow.add_node("web_search", noop)
workflow.add_node("generate", noop)
workflow.add_node("grade", noop)
workflow.add_node("generate_subquestions", noop)
workflow.add_node("multi_hop_retrieve", noop)

# Add edges
workflow.add_edge("initial_query", "retrieve")
workflow.add_edge("retrieve", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "grade")
workflow.add_conditional_edges(
    "grade", 
    always_continue,
    {
        "multi_hop_retrieve": "multi_hop_retrieve",
        "generate_subquestions": "generate_subquestions",
        "end": lg.END
    }
)
workflow.add_edge("generate_subquestions", "multi_hop_retrieve")
workflow.add_edge("multi_hop_retrieve", "generate")

# Set entry point
workflow.set_entry_point("initial_query")

# Compile graph
app = workflow.compile()

if __name__ == "__main__":
    # Print graph structure
    print("\nHybrid RAG Graph Structure:")
    print(app.get_graph().draw_mermaid()) 