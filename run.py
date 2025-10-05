#!/usr/bin/env python
"""
AstroLife Explorer - Main Pipeline Script

This script orchestrates the entire pipeline for the AstroLife Explorer project:
1. Download data from NASA bioscience corpus
2. Preprocess the data
3. Train the Sentence-BERT model
4. Create FAISS index for efficient retrieval
5. Evaluate the model
6. Start the web interface

Usage:
    python run.py --all                # Run the entire pipeline
    python run.py --download           # Only download data
    python run.py --preprocess         # Only preprocess data
    python run.py --train              # Only train model
    python run.py --index              # Only create index
    python run.py --evaluate           # Only evaluate model
    python run.py --serve              # Only start web interface
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), "astrolife_explorer.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def download_data():
    """
    Download the NASA bioscience corpus
    """
    try:
        from src.data_processing.download_data import download_dataset, load_dataset
        
        logger.info("Starting data download...")
        
        # URL to the raw CSV file on GitHub
        github_url = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"
        
        # Local path to save the dataset
        local_path = os.path.join(os.path.dirname(__file__), "data", "raw", "SB_publication_PMC.csv")
        
        # Download the dataset
        download_dataset(github_url, local_path)
        
        # Load and verify the dataset
        df = load_dataset(local_path)
        
        logger.info(f"Data download completed successfully. Downloaded {len(df)} records.")
        return True
    except Exception as e:
        logger.error(f"Error during data download: {str(e)}")
        return False

def preprocess_data():
    """
    Preprocess the NASA bioscience corpus
    """
    try:
        from src.data_processing.preprocess import preprocess_dataset
        
        logger.info("Starting data preprocessing...")
        
        # Paths
        input_path = os.path.join(os.path.dirname(__file__), "data", "raw", "SB_publication_PMC.csv")
        output_dir = os.path.join(os.path.dirname(__file__), "data", "processed")
        
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            logger.info("Please run the download step first.")
            return False
        
        # Preprocess the dataset
        result = preprocess_dataset(input_path, output_dir)
        
        if result:
            logger.info("Data preprocessing completed successfully.")
            return True
        else:
            logger.error("Data preprocessing failed.")
            return False
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        return False

def train_model():
    """
    Train the Sentence-BERT model
    """
    try:
        from src.model.train import train_model, evaluate_model
        
        logger.info("Starting model training...")
        
        # Paths
        train_data_path = os.path.join(os.path.dirname(__file__), "data", "processed", "train.json")
        val_data_path = os.path.join(os.path.dirname(__file__), "data", "processed", "val.json")
        test_data_path = os.path.join(os.path.dirname(__file__), "data", "processed", "test.json")
        output_dir = os.path.join(os.path.dirname(__file__), "models", "sbert_nasa")
        
        # Check if data files exist
        for data_path in [train_data_path, val_data_path, test_data_path]:
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                logger.info("Please run the preprocessing step first.")
                return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Train the model
        logger.info("Training model...")
        model = train_model(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            output_dir=output_dir,
            base_model="paraphrase-MiniLM-L3-v2",  # Use smaller model for faster training
            batch_size=16,
            epochs=3,  # Reduced for demonstration
            learning_rate=2e-5,
            warmup_steps=100,
            use_amp=False
        )
        
        # Quick evaluation
        logger.info("Evaluating model...")
        evaluate_model(model, test_data_path, batch_size=16)
        
        logger.info("Model training completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False

def create_index():
    """
    Create FAISS index for efficient retrieval
    """
    try:
        from src.model.indexing import FAISSIndex
        
        logger.info("Starting index creation...")
        
        # Paths
        model_path = os.path.join(os.path.dirname(__file__), "models", "sbert_nasa")
        index_dir = os.path.join(os.path.dirname(__file__), "models", "faiss_index")
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, "index.faiss")
        data_path = os.path.join(os.path.dirname(__file__), "data", "processed", "sample_train.json")
        
        # Check if model and data exist
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.info("Please run the training step first.")
            return False
        
        if not os.path.exists(data_path):
            logger.error(f"Data not found: {data_path}")
            logger.info("Please run the preprocessing step first.")
            return False
        
        # Create and build index
        logger.info("Building FAISS index...")
        faiss_index = FAISSIndex(model_path=model_path, index_path=index_path, data_path=data_path)
        faiss_index.build_index()
        
        # Test search
        query = "Effects of microgravity on human physiology"
        results = faiss_index.search(query, k=3)
        
        logger.info(f"Index creation completed successfully. Index contains {faiss_index.index.ntotal} vectors.")
        logger.info(f"Test query: '{query}'")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1} (Score: {result['score']:.4f}): {result['passage'][:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error during index creation: {str(e)}")
        return False

def evaluate():
    """
    Evaluate the model and search system
    """
    try:
        from src.evaluation.metrics import evaluate_search_system
        from src.model.indexing import FAISSIndex
        
        logger.info("Starting evaluation...")
        
        # Paths
        model_path = os.path.join(os.path.dirname(__file__), "models", "sbert_nasa")
        index_path = os.path.join(os.path.dirname(__file__), "models", "faiss_index", "index.faiss")
        test_data_path = os.path.join(os.path.dirname(__file__), "data", "processed", "test.json")
        output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
        
        # Check if model, index, and test data exist
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.info("Please run the training step first.")
            return False
        
        if not os.path.exists(index_path):
            logger.error(f"Index not found: {index_path}")
            logger.info("Please run the indexing step first.")
            return False
        
        if not os.path.exists(test_data_path):
            logger.error(f"Test data not found: {test_data_path}")
            logger.info("Please run the preprocessing step first.")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load index
        logger.info("Loading FAISS index...")
        faiss_index = FAISSIndex(model_path=model_path)
        faiss_index.load_index(index_path)
        
        # Evaluate search system
        logger.info("Evaluating search system...")
        metrics = evaluate_search_system(faiss_index, test_data_path, output_dir)
        
        logger.info("Evaluation completed successfully.")
        logger.info(f"MRR: {metrics['mrr']:.4f}")
        logger.info(f"Recall@1: {metrics['recall_at_1']:.4f}")
        logger.info(f"Recall@5: {metrics['recall_at_5']:.4f}")
        logger.info(f"Recall@10: {metrics['recall_at_10']:.4f}")
        logger.info(f"Coherence: {metrics['coherence']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return False

def serve():
    """
    Start the web interface
    """
    try:
        from src.api.app import create_app
        
        logger.info("Starting web interface...")
        
        # Paths
        model_path = os.path.join(os.path.dirname(__file__), "models", "sbert_nasa")
        index_path = os.path.join(os.path.dirname(__file__), "models", "faiss_index", "index.faiss")
        
        # Check if model and index exist
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.info("Please run the training step first.")
            return False
        
        if not os.path.exists(index_path):
            logger.error(f"Index not found: {index_path}")
            logger.info("Please run the indexing step first.")
            return False
        
        # Create and run app
        app = create_app(model_path, index_path)
        
        logger.info("Web interface started successfully.")
        logger.info("Navigate to http://localhost:5000 in your web browser to access the AstroLife Explorer.")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
        return True
    except Exception as e:
        logger.error(f"Error starting web interface: {str(e)}")
        return False

def run_all():
    """
    Run the entire pipeline
    """
    logger.info("Running the entire pipeline...")
    
    # Track overall success
    success = True
    
    # Track timing
    start_time = time.time()
    
    # Run each step
    steps = [
        ("Data Download", download_data),
        ("Data Preprocessing", preprocess_data),
        ("Model Training", train_model),
        ("Index Creation", create_index),
        ("Evaluation", evaluate)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starting step: {step_name}")
        step_start_time = time.time()
        
        step_success = step_func()
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if step_success:
            logger.info(f"{step_name} completed successfully in {step_duration:.2f} seconds.")
        else:
            logger.error(f"{step_name} failed after {step_duration:.2f} seconds.")
            success = False
            break
    
    # Start web interface if all steps succeeded
    if success:
        logger.info(f"\n{'=' * 50}")
        logger.info("Starting Web Interface")
        serve()
    
    # Calculate total duration
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"\n{'=' * 50}")
    if success:
        logger.info(f"Entire pipeline completed successfully in {total_duration:.2f} seconds.")
    else:
        logger.error(f"Pipeline failed after {total_duration:.2f} seconds.")
    
    return success

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="AstroLife Explorer Pipeline")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline")
    parser.add_argument("--download", action="store_true", help="Download the NASA bioscience corpus")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train", action="store_true", help="Train the Sentence-BERT model")
    parser.add_argument("--index", action="store_true", help="Create FAISS index")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--serve", action="store_true", help="Start the web interface")
    
    args = parser.parse_args()
    
    # Print banner
    print(f"\n{'=' * 60}")
    print(f"AstroLife Explorer - NASA Bioscience Q&A Engine")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run requested steps
    if args.all:
        run_all()
    else:
        if args.download:
            download_data()
        if args.preprocess:
            preprocess_data()
        if args.train:
            train_model()
        if args.index:
            create_index()
        if args.evaluate:
            evaluate()
        if args.serve:
            serve()

if __name__ == "__main__":
    main()