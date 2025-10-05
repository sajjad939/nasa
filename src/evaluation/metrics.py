import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def mean_reciprocal_rank(results, relevant_docs):
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        results (list): List of search results
        relevant_docs (list): List of relevant document IDs
        
    Returns:
        float: MRR score
    """
    for i, result in enumerate(results):
        if result['passage_id'] in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(results, relevant_docs, k=10):
    """
    Calculate Recall@k
    
    Args:
        results (list): List of search results
        relevant_docs (list): List of relevant document IDs
        k (int): k value
        
    Returns:
        float: Recall@k score
    """
    results_at_k = results[:k]
    retrieved_relevant = sum(1 for result in results_at_k if result['passage_id'] in relevant_docs)
    return retrieved_relevant / len(relevant_docs) if len(relevant_docs) > 0 else 0.0

def semantic_coherence(model, query, results, top_k=5):
    """
    Calculate semantic coherence between query and results
    
    Args:
        model (SentenceTransformer): Sentence transformer model
        query (str): Query text
        results (list): List of search results
        top_k (int): Number of top results to consider
        
    Returns:
        float: Average cosine similarity
    """
    if not results:
        return 0.0
    
    # Get top k results
    top_results = results[:top_k]
    passages = [result['passage'] for result in top_results]
    
    # Encode query and passages
    query_embedding = model.encode([query], convert_to_numpy=True)
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, passage_embeddings)[0]
    
    # Return average similarity
    return float(np.mean(similarities))

def evaluate_search_system(model, index, test_data_path, output_dir=None):
    """
    Evaluate the search system on test data
    
    Args:
        model (SentenceTransformer): Sentence transformer model
        index: FAISS index object
        test_data_path (str): Path to test data
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation results
    """
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize metrics
    mrr_scores = []
    recall_at_1_scores = []
    recall_at_5_scores = []
    recall_at_10_scores = []
    coherence_scores = []
    
    # Evaluate each query
    print(f"Evaluating on {len(test_data)} test queries")
    for item in tqdm(test_data):
        query = item['query']
        relevant_doc_id = f"{item['pmcid']}_{item['segment_id']}"
        
        # Search
        results = index.search(query, k=10)
        
        # Calculate metrics
        mrr = mean_reciprocal_rank(results, [relevant_doc_id])
        r_at_1 = recall_at_k(results, [relevant_doc_id], k=1)
        r_at_5 = recall_at_k(results, [relevant_doc_id], k=5)
        r_at_10 = recall_at_k(results, [relevant_doc_id], k=10)
        coherence = semantic_coherence(model, query, results)
        
        # Append scores
        mrr_scores.append(mrr)
        recall_at_1_scores.append(r_at_1)
        recall_at_5_scores.append(r_at_5)
        recall_at_10_scores.append(r_at_10)
        coherence_scores.append(coherence)
    
    # Calculate average metrics
    avg_mrr = np.mean(mrr_scores)
    avg_recall_at_1 = np.mean(recall_at_1_scores)
    avg_recall_at_5 = np.mean(recall_at_5_scores)
    avg_recall_at_10 = np.mean(recall_at_10_scores)
    avg_coherence = np.mean(coherence_scores)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
    print(f"Recall@1: {avg_recall_at_1:.4f}")
    print(f"Recall@5: {avg_recall_at_5:.4f}")
    print(f"Recall@10: {avg_recall_at_10:.4f}")
    print(f"Semantic Coherence: {avg_coherence:.4f}")
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics = {
            'mrr': avg_mrr,
            'recall_at_1': avg_recall_at_1,
            'recall_at_5': avg_recall_at_5,
            'recall_at_10': avg_recall_at_10,
            'coherence': avg_coherence
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot metrics
        plt.figure(figsize=(10, 6))
        metrics_names = ['MRR', 'Recall@1', 'Recall@5', 'Recall@10', 'Coherence']
        metrics_values = [avg_mrr, avg_recall_at_1, avg_recall_at_5, avg_recall_at_10, avg_coherence]
        plt.bar(metrics_names, metrics_values)
        plt.ylim(0, 1)
        plt.title('Search System Evaluation Metrics')
        plt.savefig(os.path.join(output_dir, 'metrics.png'))
        
        print(f"Evaluation results saved to {output_dir}")
    
    return {
        'mrr': avg_mrr,
        'recall_at_1': avg_recall_at_1,
        'recall_at_5': avg_recall_at_5,
        'recall_at_10': avg_recall_at_10,
        'coherence': avg_coherence
    }

if __name__ == "__main__":
    # This is a placeholder for demonstration
    # In practice, you would load your model and index
    # and run the evaluation on your test data
    print("This module provides evaluation metrics for the search system.")
    print("Import and use the functions in your evaluation script.")