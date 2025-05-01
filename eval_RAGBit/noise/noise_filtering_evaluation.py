#!/usr/bin/env python
# File: eval_RAGBit/noise_filtering_evaluation.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from hybrid_rag import retrieve, embedding, vectorstore
from eval_RAGBit.test_functions import analyze_retrieval_metrics
import json
import numpy as np
from langchain_core.documents import Document
import traceback

def run_noise_filtering_evaluation():
    """
    Evaluates the system's noise filtering capabilities by running a series of test queries
    and analyzing the results.
    """
    print("Running Noise Filtering Evaluation...")
    
    # Test queries covering different domains and complexity levels
    test_queries = [
        "What is the average delivery time by country?",
        "What factors explain the high return rate in China?",
        "How has delivery time evolved in Spain over time?",
        "Which countries have the lowest average user age and why?",
        "What is the correlation between distance and delivery speed?",
        "How do weather conditions affect delivery times?",
        "What is the impact of holidays on delivery performance?",
        "How does the size of the delivery fleet affect delivery times?",
        "What is the relationship between customer satisfaction and delivery time?",
        "How do different payment methods affect delivery times?"
    ]
    
    # Run each query through the retrieval process
    for i, query in enumerate(test_queries):
        print(f"\nProcessing query {i+1}/{len(test_queries)}: {query}")
        
        # Initialize state with the test query
        state = {"question": query, "documents": []}
        
        # Run the retrieval process
        try:
            retrieved_state = retrieve(state)
            print(f"Retrieved {len(retrieved_state['documents'])} documents")
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")
            traceback.print_exc()
    
    # Analyze the collected metrics
    print("\nAnalyzing retrieval metrics...")
    analysis = analyze_retrieval_metrics()
    
    if analysis:
        print("\nAnalysis Results:")
        print(f"Total Queries Analyzed: {analysis['total_queries']}")
        print(f"Average Initial Documents: {analysis['avg_initial_count']}")
        print(f"Average After Similarity Filtering: {analysis['avg_filtered_count']}")
        print(f"Average After Deduplication: {analysis['avg_deduped_count']}")
        print(f"Average Filtering Rate: {analysis['avg_filtering_rate']}%")
        print(f"Average Deduplication Rate: {analysis['avg_dedup_rate']}%")
        print(f"Average Total Reduction: {analysis['avg_total_reduction']}%")
    else:
        print("No analysis results available.")

def test_with_synthetic_data():
    """
    Tests the noise filtering mechanisms with synthetic data.
    This allows us to evaluate the filtering mechanisms without relying on the actual vectorstore.
    """
    print("\nTesting with synthetic data...")
    
    # Create synthetic documents with known duplicates
    synthetic_docs = [
        Document(page_content="The average delivery time in Spain is 48 hours."),
        Document(page_content="The average delivery time in France is 52 hours."),
        Document(page_content="The return rate in China is 15%."),
        Document(page_content="The user age in the US is 32 years."),
        Document(page_content="The distance from warehouse to customer in Spain is 500km."),
        Document(page_content="The delivery time in Spain averages 48 hours."),  # Duplicate of first doc
        Document(page_content="China has a return rate of 15%."),  # Duplicate of third doc
        Document(page_content="The average delivery time in Germany is 55 hours."),
        Document(page_content="The return rate in Japan is 10%."),
        Document(page_content="The user age in the UK is 35 years.")
    ]
    
    # Add synthetic documents to the vectorstore
    print("Adding synthetic documents to vectorstore...")
    try:
        vectorstore.add_documents(synthetic_docs)
        print(f"Added {len(synthetic_docs)} synthetic documents to vectorstore")
    except Exception as e:
        print(f"Error adding synthetic documents: {str(e)}")
        return
    
    # Test queries
    test_queries = [
        "What is the delivery time in Spain?",
        "What is the return rate in China?",
        "What is the user age in the US?"
    ]
    
    # Run each query through the retrieval process
    for i, query in enumerate(test_queries):
        print(f"\nProcessing synthetic query {i+1}/{len(test_queries)}: {query}")
        
        # Initialize state with the test query
        state = {"question": query, "documents": []}
        
        # Run the retrieval process
        try:
            retrieved_state = retrieve(state)
            print(f"Retrieved {len(retrieved_state['documents'])} documents")
        except Exception as e:
            print(f"Error processing synthetic query '{query}': {str(e)}")
            traceback.print_exc()

def test_with_different_thresholds():
    """
    Tests the noise filtering mechanisms with different similarity and deduplication thresholds.
    """
    print("\nTesting with different thresholds...")
    
    # Define test thresholds
    similarity_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    dedup_thresholds = [0.8, 0.85, 0.9, 0.95, 0.98]
    
    # Test query
    query = "What is the average delivery time by country?"
    
    # Initialize state with the test query
    state = {"question": query, "documents": []}
    
    # Run the retrieval process with different thresholds
    for sim_threshold in similarity_thresholds:
        for dedup_threshold in dedup_thresholds:
            print(f"\nTesting with similarity_threshold={sim_threshold}, dedup_threshold={dedup_threshold}")
            
            # Modify the retrieve function to use the test thresholds
            # This is a bit of a hack, but it's the simplest way to test different thresholds
            from eval_RAGBit.test_functions import track_retrieval_metrics
            
            # Run the retrieval process
            try:
                # Get initial documents
                rewritten_question = state["question"]
                candidate_docs = vectorstore.similarity_search(rewritten_question, k=15)
                candidate_texts = [doc.page_content for doc in candidate_docs]
                
                # Create embeddings for the documents and query
                query_emb = embedding.embed_query(rewritten_question)
                doc_embs = [embedding.embed_query(d) for d in candidate_texts]
                
                # Filter by similarity
                filtered_docs = []
                filtered_embs = []
                for doc, emb in zip(candidate_texts, doc_embs):
                    sim = np.dot(emb, query_emb) / (np.linalg.norm(emb) * np.linalg.norm(query_emb))
                    if sim >= sim_threshold:
                        filtered_docs.append(doc)
                        filtered_embs.append(emb)
                
                # Deduplicate documents
                used = [False] * len(filtered_docs)
                deduped_docs = []
                deduped_embs = []
                for i, emb_i in enumerate(filtered_embs):
                    if used[i]:
                        continue
                    # Keep this doc
                    deduped_docs.append(filtered_docs[i])
                    deduped_embs.append(filtered_embs[i])
                    for j in range(i + 1, len(filtered_embs)):
                        if not used[j]:
                            sim = np.dot(emb_i, filtered_embs[j]) / (np.linalg.norm(emb_i) * np.linalg.norm(filtered_embs[j]))
                            if sim >= dedup_threshold:
                                used[j] = True
                
                # Track metrics
                track_retrieval_metrics(
                    initial_docs=candidate_texts,
                    filtered_docs=filtered_docs,
                    deduped_docs=deduped_docs,
                    query=query,
                    similarity_threshold=sim_threshold,
                    dedup_threshold=dedup_threshold
                )
                
                print(f"Initial docs: {len(candidate_texts)}, Filtered docs: {len(filtered_docs)}, Deduped docs: {len(deduped_docs)}")
            except Exception as e:
                print(f"Error testing thresholds: {str(e)}")
                traceback.print_exc()

if __name__ == "__main__":
    try:
        # Run the main evaluation
        run_noise_filtering_evaluation()
        
        # Test with synthetic data
        test_with_synthetic_data()
        
        # Test with different thresholds
        test_with_different_thresholds()
        
        # Analyze the collected metrics
        print("\nAnalyzing all retrieval metrics...")
        analysis = analyze_retrieval_metrics()
        
        if analysis:
            print("\nFinal Analysis Results:")
            print(f"Total Queries Analyzed: {analysis['total_queries']}")
            print(f"Average Initial Documents: {analysis['avg_initial_count']}")
            print(f"Average After Similarity Filtering: {analysis['avg_filtered_count']}")
            print(f"Average After Deduplication: {analysis['avg_deduped_count']}")
            print(f"Average Filtering Rate: {analysis['avg_filtering_rate']}%")
            print(f"Average Deduplication Rate: {analysis['avg_dedup_rate']}%")
            print(f"Average Total Reduction: {analysis['avg_total_reduction']}%")
        else:
            print("No analysis results available.")
    except Exception as e:
        print(f"Error running evaluation: {str(e)}")
        traceback.print_exc() 