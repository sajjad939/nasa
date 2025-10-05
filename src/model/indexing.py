import os
import json
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class FAISSIndex:
    """
    FAISS vector index for efficient similarity search
    """
    def __init__(self, model_path=None, index_path=None, data_path=None):
        """
        Initialize the FAISS index
        
        Args:
            model_path (str): Path to the Sentence-BERT model
            index_path (str): Path to save/load the FAISS index
            data_path (str): Path to the data for indexing
        """
        self.model_path = model_path
        self.index_path = index_path
        self.data_path = data_path
        self.model = None
        self.index = None
        self.passages = []
        self.passage_ids = []
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the Sentence-BERT model
        
        Args:
            model_path (str): Path to the model
        """
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        print("Model loaded successfully")
    
    def build_index(self, data_path=None, use_gpu=False):
        """
        Build the FAISS index from data
        
        Args:
            data_path (str): Path to the data for indexing
            use_gpu (bool): Whether to use GPU for indexing
            
        Returns:
            faiss.Index: Built index
        """
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            raise ValueError("Data path must be provided")
        
        if not self.model:
            raise ValueError("Model must be loaded before building index")
        
        # Load data
        print(f"Loading data from {self.data_path}")
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            # Extract passages from abstract
            self.passages = df['Abstract'].fillna('').tolist()
            self.passage_ids = df['PMCID'].astype(str).tolist()
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            # Extract passages from JSON data
            self.passages = [item['positive'] for item in data]
            self.passage_ids = [f"{item['pmcid']}_{item['segment_id']}" for item in data]
        
        print(f"Loaded {len(self.passages)} passages")
        
        # Encode passages
        print("Encoding passages...")
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(self.passages), batch_size)):
            batch = self.passages[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"Encoded {embeddings.shape[0]} passages with dimension {embeddings.shape[1]}")
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Create index
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for indexing")
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, dimension, config)
        else:
            print("Using CPU for indexing")
            # Use HNSW index for better performance
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
            index.hnsw.efConstruction = 200  # Higher value = better quality but slower build
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        index.add(embeddings)
        print(f"Added {index.ntotal} vectors to index")
        
        self.index = index
        
        # Save index if path is provided
        if self.index_path:
            self.save_index(self.index_path)
        
        return index
    
    def save_index(self, index_path=None):
        """
        Save the FAISS index
        
        Args:
            index_path (str): Path to save the index
        """
        if index_path:
            self.index_path = index_path
        
        if not self.index_path:
            raise ValueError("Index path must be provided")
        
        if not self.index:
            raise ValueError("Index must be built before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save index
        print(f"Saving index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        
        # Save passages and IDs
        metadata_path = self.index_path + ".metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'passages': self.passages,
                'passage_ids': self.passage_ids
            }, f)
        
        print(f"Index and metadata saved successfully")
    
    def load_index(self, index_path=None):
        """
        Load the FAISS index
        
        Args:
            index_path (str): Path to load the index from
        """
        if index_path:
            self.index_path = index_path
        
        if not self.index_path:
            raise ValueError("Index path must be provided")
        
        # Load index
        print(f"Loading index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"Loaded index with {self.index.ntotal} vectors")
        
        # Load passages and IDs
        metadata_path = self.index_path + ".metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.passages = metadata['passages']
            self.passage_ids = metadata['passage_ids']
        
        print(f"Loaded {len(self.passages)} passages")
    
    def search(self, query, k=10):
        """
        Search the index for similar passages
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            list: List of dictionaries with search results
        """
        if not self.index:
            raise ValueError("Index must be loaded before searching")
        
        if not self.model:
            raise ValueError("Model must be loaded before searching")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.passages):
                results.append({
                    'passage': self.passages[idx],
                    'passage_id': self.passage_ids[idx],
                    'score': float(scores[0][i])
                })
        
        return results

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "models", "sbert_nasa")
    index_path = os.path.join(base_dir, "models", "faiss_index")
    data_path = os.path.join(base_dir, "data", "processed", "train.json")
    
    # Create and build index
    faiss_index = FAISSIndex(model_path=model_path, index_path=index_path, data_path=data_path)
    faiss_index.build_index()
    
    # Test search
    query = "Effects of microgravity on human physiology"
    results = faiss_index.search(query, k=5)
    
    print(f"\nSearch results for query: '{query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f})")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Passage: {result['passage'][:200]}...\n")