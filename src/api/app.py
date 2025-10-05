import os
import json
from flask import Flask, request, jsonify, render_template
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.indexing import FAISSIndex

app = Flask(__name__)

# Global variables
model_path = None
index_path = None
faiss_index = None

def initialize_search_engine(model_dir, index_file):
    """
    Initialize the search engine with model and index
    
    Args:
        model_dir (str): Path to the model directory
        index_file (str): Path to the index file
    """
    global model_path, index_path, faiss_index
    
    model_path = model_dir
    index_path = index_file
    
    # Initialize FAISS index
    print(f"Initializing search engine with model: {model_path} and index: {index_path}")
    faiss_index = FAISSIndex(model_path=model_path)
    faiss_index.load_index(index_path)
    print("Search engine initialized successfully")

@app.route('/')
def home():
    """
    Home page
    """
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """
    Search API endpoint
    """
    # Get request data
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 10)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Search
    try:
        results = faiss_index.search(query, k=k)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """
    Health check endpoint
    """
    if faiss_index and faiss_index.model and faiss_index.index:
        return jsonify({'status': 'healthy'})
    else:
        return jsonify({'status': 'unhealthy'}), 500

def create_app(model_dir=None, index_file=None):
    """
    Create and configure the Flask app
    
    Args:
        model_dir (str): Path to the model directory
        index_file (str): Path to the index file
        
    Returns:
        Flask: Configured Flask app
    """
    if model_dir and index_file:
        initialize_search_engine(model_dir, index_file)
    
    return app

if __name__ == '__main__':
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_dir = os.path.join(base_dir, "models", "sbert_nasa")
    index_file = os.path.join(base_dir, "models", "faiss_index")
    
    # Create app
    app = create_app(model_dir, index_file)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)