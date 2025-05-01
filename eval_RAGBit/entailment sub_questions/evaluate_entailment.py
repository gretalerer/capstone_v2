# %%
"""
Entailment Evaluation Script for Multi-hop Questioning

This script evaluates how well the generated subquestions address the identified gaps
in the causal explanation using a DeBERTa entailment model.
"""

import json
import os
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob
import re
# %%

# Load the gap analysis history
def load_gap_analysis_history(file_path):
    """Load the gap analysis history from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

# Load the DeBERTa model and tokenizer
def load_entailment_model(model_name="microsoft/deberta-v3-large-mnli"):
    """Load the DeBERTa model and tokenizer for entailment classification.
    
    This function loads a model fine-tuned for the MNLI (Multi-Genre Natural Language Inference) task,
    which is suitable for evaluating entailment. The model should have 3 output classes:
    0: contradiction, 1: neutral, 2: entailment.
    """
    try:
        # Try using the DeBERTaV2Tokenizer directly instead of AutoTokenizer
        from transformers import DebertaV2Tokenizer
        
        print(f"Attempting to load {model_name}...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Verify the model has the correct number of output classes for MNLI
        num_labels = model.config.num_labels
        if num_labels != 3:
            raise ValueError(f"Model has {num_labels} output classes, but we need exactly 3 for entailment classification (contradiction, neutral, entailment)")
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading DeBERTa model: {e}")
        print(f"Attempting with alternate model...")
        
        # Try with a RoBERTa model which has fewer tokenizer issues
        alt_model_name = "roberta-large-mnli"
        try:
            print(f"Loading {alt_model_name} instead...")
            tokenizer = AutoTokenizer.from_pretrained(alt_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(alt_model_name)
            
            # Verify the model has the correct number of output classes for MNLI
            num_labels = model.config.num_labels
            if num_labels != 3:
                raise ValueError(f"Model has {num_labels} output classes, but we need exactly 3 for entailment classification")
                
            return model, tokenizer
        except Exception as e2:
            print(f"Error loading alternate model: {e2}")
            raise ValueError(f"Failed to load any suitable models. Both models must be fine-tuned for MNLI with 3 output classes.\nOriginal error: {e}\nAlternate model error: {e2}")

# %%
# Evaluate entailment between a gap analysis and a question
def evaluate_entailment(model, tokenizer, gap_analysis, question):
    """
    Evaluate the entailment between a framed gap analysis and a subquestion.

    Args:
        model: The entailment model (e.g., DeBERTa or RoBERTa)
        tokenizer: The associated tokenizer
        gap_analysis: The gap analysis text (premise)
        question: The subquestion to test against the gap (hypothesis)

    Returns:
        entailment_score (float): Likelihood that the subquestion addresses the gap
    """

    # Frame the gap to give the model more context
    framed_gap = (
        "We have identified the following explanatory gaps in our analysis:\n"
        f"{gap_analysis}\n\n"
        "Does the following subquestion help clarify or address any of these gaps?"
    )

    # Prepare input pair: (premise, hypothesis)
    inputs = tokenizer(
        framed_gap,
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        entailment_score = probabilities[0][2].item()  # Index 2 = entailment

    return entailment_score


# Evaluate all gap analyses and their corresponding subquestions
def evaluate_all_entailments(gap_analysis_history, model, tokenizer):
    """
    Evaluate all gap analyses and their corresponding subquestions:
        gap_analysis_history: Dictionary mapping gap analyses to lists of subquestions
        model: The DeBERTa model
        tokenizer: The tokenizer for the model
        
    Returns:
        results: Dictionary mapping gap analyses to dictionaries mapping subquestions to entailment scores
    """
    results = {}
    
    for gap_analysis, subquestions in tqdm(gap_analysis_history.items(), desc="Evaluating entailments"):
        gap_results = {}
        
        for question in subquestions:
            # Clean up the question (remove numbering if present)
            clean_question = question
            if question.startswith(("1.", "2.", "3.")):
                clean_question = question[2:].strip()
            
            # Evaluate entailment
            entailment_score = evaluate_entailment(model, tokenizer, gap_analysis, clean_question)
            gap_results[question] = entailment_score
        
        results[gap_analysis] = gap_results
    
    return results

# %%

# Calculate statistics for the entailment scores
def calculate_statistics(results):
    """
    Calculate statistics for the entailment scores.
    
    Args:
        results: Dictionary mapping gap analyses to dictionaries mapping subquestions to entailment scores
        
    Returns:
        stats: Dictionary containing statistics about the entailment scores
    """
    all_scores = []
    gap_avg_scores = []
    
    for gap_analysis, gap_results in results.items():
        scores = list(gap_results.values())
        all_scores.extend(scores)
        gap_avg_scores.append(np.mean(scores))
    
    stats = {
        "mean_score": np.mean(all_scores),
        "median_score": np.median(all_scores),
        "std_score": np.std(all_scores),
        "min_score": np.min(all_scores),
        "max_score": np.max(all_scores),
        "mean_gap_score": np.mean(gap_avg_scores),
        "median_gap_score": np.median(gap_avg_scores),
        "std_gap_score": np.std(gap_avg_scores),
        "min_gap_score": np.min(gap_avg_scores),
        "max_gap_score": np.max(gap_avg_scores)
    }
    
    return stats

# %%

# Visualize the entailment scores
def visualize_entailment_scores(results, output_dir=".", stats=None, output_filename="entailment_scores.png"):
    """
    Visualize the entailment scores with summary statistics included as a table.

    Args:
        results: Dictionary mapping gap analyses to dictionaries mapping subquestions to entailment scores
        output_dir: Directory to save the visualizations
        stats: Optional dictionary of summary statistics to annotate in the figure
        output_filename: Name of the output file
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle("Entailment Score Evaluation Summary", fontsize=16)
    axes = axes.flatten()

    # Plot 1: Histogram
    all_scores = [s for gap_results in results.values() for s in gap_results.values()]
    sns.histplot(all_scores, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Entailment Scores")
    axes[0].set_xlabel("Entailment Score")
    axes[0].set_ylabel("Count")

    # Plot 2: Box plot
    gap_avg_scores = []
    gap_labels = []
    for i, (gap_analysis, gap_results) in enumerate(results.items()):
        scores = list(gap_results.values())
        gap_avg_scores.append(scores)
        gap_labels.append(f"Gap {i+1}")

    sns.boxplot(data=gap_avg_scores, ax=axes[1])
    axes[1].set_title("Entailment Scores by Gap Analysis")
    axes[1].set_xlabel("Gap Analysis")
    axes[1].set_ylabel("Entailment Score")
    axes[1].set_xticklabels(gap_labels, rotation=45)

    # Plot 3: Heatmap
    max_questions = max(len(gap_results) for gap_results in results.values())
    score_matrix = np.zeros((len(results), max_questions))
    for i, gap_results in enumerate(results.values()):
        scores = list(gap_results.values())
        score_matrix[i, :len(scores)] = scores
    sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[2])
    axes[2].set_title("Heatmap of Entailment Scores")
    axes[2].set_xlabel("Question Index")
    axes[2].set_ylabel("Gap Analysis Index")

    # Plot 4: Bar chart
    gap_means = [np.mean(scores) for scores in gap_avg_scores]
    sns.barplot(x=gap_labels, y=gap_means, ax=axes[3])
    axes[3].set_title("Average Entailment Score by Gap Analysis")
    axes[3].set_xlabel("Gap Analysis")
    axes[3].set_ylabel("Average Entailment Score")
    axes[3].set_xticklabels(gap_labels, rotation=45)

    # Plot 5: Summary statistics table
    axes[4].axis("off")
    if stats:
        table_data = [
            ["Metric", "Value"],
            ["Mean Entailment", f"{stats['mean_score']:.4f}"],
            ["Median Entailment", f"{stats['median_score']:.4f}"],
            ["Std Dev", f"{stats['std_score']:.4f}"],
            ["Min Score", f"{stats['min_score']:.4f}"],
            ["Max Score", f"{stats['max_score']:.4f}"],
            ["Mean Gap Score", f"{stats['mean_gap_score']:.4f}"],
            ["Median Gap Score", f"{stats['median_gap_score']:.4f}"]
        ]
        table = axes[4].table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[4].set_title("Summary Statistics", pad=20)

    # Hide unused subplot 6
    axes[5].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()

# %%

# Save the results to a JSON file
def save_results(results, stats, output_file):
    """
    Save the results to a JSON file:
        results: Dictionary mapping gap analyses to dictionaries mapping subquestions to entailment scores
        stats: Dictionary containing statistics about the entailment scores
        output_file: Path to save the results
    """
    output = {
        "results": results,
        "statistics": stats
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

# Process a single gap analysis history file
def process_gap_analysis_file(file_path, model, tokenizer, output_dir="."):
    """
    Process a single gap analysis history file and generate visualizations.
    
    Args:
        file_path: Path to the gap analysis history file
        model: The entailment model
        tokenizer: The tokenizer for the model
        output_dir: Directory to save the results
    """
    # Extract the file number from the filename
    match = re.search(r'gap_analysis_history_(\d+)\.json', file_path)
    file_number = match.group(1) if match else "unknown"
    
    print(f"\n{'='*80}")
    print(f"Processing file: {file_path}")
    print(f"{'='*80}\n")
    
    # Load the gap analysis history
    gap_analysis_history = load_gap_analysis_history(file_path)
    print(f"Loaded {len(gap_analysis_history)} gap analyses from {file_path}")
    
    # Evaluate all entailments
    print("Evaluating entailments...")
    results = evaluate_all_entailments(gap_analysis_history, model, tokenizer)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(results)
    
    # Visualize the entailment scores
    print("Visualizing entailment scores...")
    output_filename = f"entailment_scores_{file_number}.png"
    visualize_entailment_scores(results, output_dir=output_dir, stats=stats, output_filename=output_filename)
    
    # Save the results
    print("Saving results...")
    output_file = f"entailment_results_{file_number}.json"
    save_results(results, stats, os.path.join(output_dir, output_file))
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean Entailment Score: {stats['mean_score']:.4f}")
    print(f"Median Entailment Score: {stats['median_score']:.4f}")
    print(f"Standard Deviation: {stats['std_score']:.4f}")
    print(f"Min Score: {stats['min_score']:.4f}")
    print(f"Max Score: {stats['max_score']:.4f}")
    print(f"Mean Gap Score: {stats['mean_gap_score']:.4f}")
    print(f"Median Gap Score: {stats['median_gap_score']:.4f}")
    
    # Identify the best and worst performing gaps
    gap_avg_scores = []
    gap_analyses = []
    
    for gap_analysis, gap_results in results.items():
        scores = list(gap_results.values())
        gap_avg_scores.append(np.mean(scores))
        gap_analyses.append(gap_analysis)
    
    # Sort by average score
    sorted_indices = np.argsort(gap_avg_scores)
    
    print("\nBest Performing Gap Analysis:")
    best_idx = sorted_indices[-1]
    print(f"Score: {gap_avg_scores[best_idx]:.4f}")
    print(f"Gap Analysis: {gap_analyses[best_idx][:100]}...")
    
    print("\nWorst Performing Gap Analysis:")
    worst_idx = sorted_indices[0]
    print(f"Score: {gap_avg_scores[worst_idx]:.4f}")
    print(f"Gap Analysis: {gap_analyses[worst_idx][:100]}...")
    
    return stats

# Main function
# %%
def main():
    """Main function to run the entailment evaluation on all gap analysis history files."""
    # Create output directory if it doesn't exist
    output_dir = "eval_RAGBit/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all gap analysis history files
    gap_analysis_dir = "eval_RAGBit/gap_analysis_history"
    gap_analysis_files = glob.glob(os.path.join(gap_analysis_dir, "gap_analysis_history_*.json"))
    
    if not gap_analysis_files:
        print(f"No gap analysis history files found in {gap_analysis_dir}")
        return
    
    print(f"Found {len(gap_analysis_files)} gap analysis history files")
    
    # Load the DeBERTa model and tokenizer (only once)
    print("Loading DeBERTa model and tokenizer...")
    model, tokenizer = load_entailment_model()
    
    # Process each file
    all_stats = {}
    for file_path in gap_analysis_files:
        file_name = os.path.basename(file_path)
        match = re.search(r'gap_analysis_history_(\d+)\.json', file_name)
        file_number = match.group(1) if match else "unknown"
        
        stats = process_gap_analysis_file(file_path, model, tokenizer, output_dir)
        all_stats[file_number] = stats
    
    # Create a summary of all results
    print("\n\n" + "="*80)
    print("SUMMARY OF ALL RESULTS")
    print("="*80)
    
    for file_number, stats in all_stats.items():
        print(f"\nFile: gap_analysis_history_{file_number}.json")
        print(f"Mean Entailment Score: {stats['mean_score']:.4f}")
        print(f"Median Entailment Score: {stats['median_score']:.4f}")
        print(f"Standard Deviation: {stats['std_score']:.4f}")
        print(f"Min Score: {stats['min_score']:.4f}")
        print(f"Max Score: {stats['max_score']:.4f}")
        print(f"Mean Gap Score: {stats['mean_gap_score']:.4f}")
        print(f"Median Gap Score: {stats['median_gap_score']:.4f}")
    
    print("\nAll evaluations completed successfully!")

# %%
if __name__ == "__main__":
    main() 
# %%
