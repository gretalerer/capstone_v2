import os
import json
import numpy as np
from datetime import datetime

# Initialize the global gap analysis history dictionary
gap_analysis_history = {}

# Function to save gap analysis history to JSON file
def save_gap_analysis_history():
    # Ensure the eval_RAGBit directory exists
    os.makedirs("eval_RAGBit", exist_ok=True)
    
    # Use a fixed filename for all gap analyses
    filename = "gap_analysis_history_4.json"
    
    # Check if the file already exists and load existing data
    file_path = f"eval_RAGBit/{filename}"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
                # Merge existing data with current data
                for key, value in existing_data.items():
                    if key in gap_analysis_history:
                        # Combine lists if key exists in both
                        gap_analysis_history[key].extend(value)
                    else:
                        # Add key-value pair if it doesn't exist in current data
                        gap_analysis_history[key] = value
        except json.JSONDecodeError:
            print(f"[HISTORY] Error reading existing file. Starting fresh.")
    
    # Save to JSON file
    with open(file_path, "w") as f:
        json.dump(gap_analysis_history, f, indent=2)
    print(f"[HISTORY] Saved gap analysis history with {len(gap_analysis_history)} entries to {filename}")

# Function to track retrieval metrics for noise filtering analysis
def track_retrieval_metrics(initial_docs, filtered_docs, deduped_docs, query, similarity_threshold=0.6, dedup_threshold=0.95):
    """
    Tracks and analyzes the noise filtering capabilities of the retrieval system.
    
    Args:
        initial_docs: List of documents initially retrieved from the vectorstore
        filtered_docs: List of documents after similarity filtering
        deduped_docs: List of documents after deduplication
        query: The query used for retrieval
        similarity_threshold: The threshold used for similarity filtering
        dedup_threshold: The threshold used for deduplication
    
    Returns:
        A dictionary containing metrics about the filtering process
    """
    # Calculate metrics
    initial_count = len(initial_docs)
    filtered_count = len(filtered_docs)
    deduped_count = len(deduped_docs)
    
    # Calculate filtering and deduplication rates
    filtering_rate = (initial_count - filtered_count) / initial_count * 100 if initial_count > 0 else 0
    dedup_rate = (filtered_count - deduped_count) / filtered_count * 100 if filtered_count > 0 else 0
    total_reduction = (initial_count - deduped_count) / initial_count * 100 if initial_count > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "initial_count": initial_count,
        "filtered_count": filtered_count,
        "deduped_count": deduped_count,
        "filtering_rate": round(filtering_rate, 2),
        "dedup_rate": round(dedup_rate, 2),
        "total_reduction": round(total_reduction, 2),
        "similarity_threshold": similarity_threshold,
        "dedup_threshold": dedup_threshold
    }
    
    # Save metrics to file
    save_retrieval_metrics(metrics)
    
    return metrics

def save_retrieval_metrics(metrics):
    """
    Saves retrieval metrics to a JSON file for later analysis.
    
    Args:
        metrics: Dictionary containing retrieval metrics
    """
    # Ensure the eval_RAGBit directory exists
    os.makedirs("eval_RAGBit", exist_ok=True)
    
    # Use a fixed filename for all retrieval metrics
    filename = "retrieval_metrics.json"
    file_path = f"eval_RAGBit/{filename}"
    
    # Load existing metrics if file exists
    existing_metrics = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                existing_metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"[METRICS] Error reading existing file. Starting fresh.")
    
    # Add new metrics
    existing_metrics.append(metrics)
    
    # Save to JSON file
    with open(file_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    
    print(f"[METRICS] Saved retrieval metrics to {filename}")

def analyze_retrieval_metrics():
    """
    Analyzes the saved retrieval metrics to provide insights into the noise filtering capabilities.
    
    Returns:
        A dictionary containing analysis results
    """
    # Ensure the eval_RAGBit directory exists
    os.makedirs("eval_RAGBit", exist_ok=True)
    
    # Use a fixed filename for all retrieval metrics
    filename = "retrieval_metrics.json"
    file_path = f"eval_RAGBit/{filename}"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"[ANALYSIS] No metrics file found at {file_path}")
        return None
    
    # Load metrics
    try:
        with open(file_path, "r") as f:
            metrics_list = json.load(f)
    except json.JSONDecodeError:
        print(f"[ANALYSIS] Error reading metrics file. Cannot analyze.")
        return None
    
    if not metrics_list:
        print(f"[ANALYSIS] No metrics found in file.")
        return None
    
    # Calculate average metrics
    avg_initial_count = np.mean([m["initial_count"] for m in metrics_list])
    avg_filtered_count = np.mean([m["filtered_count"] for m in metrics_list])
    avg_deduped_count = np.mean([m["deduped_count"] for m in metrics_list])
    avg_filtering_rate = np.mean([m["filtering_rate"] for m in metrics_list])
    avg_dedup_rate = np.mean([m["dedup_rate"] for m in metrics_list])
    avg_total_reduction = np.mean([m["total_reduction"] for m in metrics_list])
    
    # Create analysis dictionary
    analysis = {
        "total_queries": len(metrics_list),
        "avg_initial_count": round(avg_initial_count, 2),
        "avg_filtered_count": round(avg_filtered_count, 2),
        "avg_deduped_count": round(avg_deduped_count, 2),
        "avg_filtering_rate": round(avg_filtering_rate, 2),
        "avg_dedup_rate": round(avg_dedup_rate, 2),
        "avg_total_reduction": round(avg_total_reduction, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save analysis to file
    save_retrieval_analysis(analysis)
    
    return analysis

def save_retrieval_analysis(analysis):
    """
    Saves retrieval analysis to a JSON file.
    
    Args:
        analysis: Dictionary containing analysis results
    """
    # Ensure the eval_RAGBit directory exists
    os.makedirs("eval_RAGBit", exist_ok=True)
    
    # Use a fixed filename for the analysis
    filename = "retrieval_analysis.json"
    file_path = f"eval_RAGBit/{filename}"
    
    # Save to JSON file
    with open(file_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"[ANALYSIS] Saved retrieval analysis to {filename}")
    
    # Also save as markdown for easier reading
    markdown_path = f"eval_RAGBit/retrieval_analysis.md"
    with open(markdown_path, "w") as f:
        f.write("# Retrieval Noise Filtering Analysis\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Queries Analyzed**: {analysis['total_queries']}\n\n")
        f.write("## Average Metrics\n\n")
        f.write(f"- **Initial Documents**: {analysis['avg_initial_count']}\n")
        f.write(f"- **After Similarity Filtering**: {analysis['avg_filtered_count']}\n")
        f.write(f"- **After Deduplication**: {analysis['avg_deduped_count']}\n")
        f.write(f"- **Filtering Rate**: {analysis['avg_filtering_rate']}%\n")
        f.write(f"- **Deduplication Rate**: {analysis['avg_dedup_rate']}%\n")
        f.write(f"- **Total Reduction**: {analysis['avg_total_reduction']}%\n\n")
        f.write("## Interpretation\n\n")
        f.write("This analysis provides insights into the noise filtering capabilities of the retrieval system.\n")
        f.write("- The filtering rate indicates how many documents were removed due to low similarity to the query.\n")
        f.write("- The deduplication rate indicates how many near-duplicate documents were removed.\n")
        f.write("- The total reduction indicates the overall effectiveness of the noise filtering process.\n")
    
    print(f"[ANALYSIS] Saved retrieval analysis to {markdown_path}")
