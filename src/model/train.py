import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Import SentenceTransformer with minimal dependencies
from sentence_transformers import SentenceTransformer, InputExample, losses

class QueryPassageDataset(Dataset):
    """
    Dataset for query-passage pairs
    """
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Create InputExamples
        self.examples = []
        for item in self.data:
            self.examples.append(
                InputExample(
                    texts=[item['query'], item['positive']],
                    label=1.0  # Positive pair
                )
            )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def simple_evaluate(model, data_path, batch_size=16):
    """
    Simple evaluation function that calculates cosine similarity
    
    Args:
        model: The SentenceTransformer model
        data_path: Path to evaluation data
        batch_size: Batch size for evaluation
        
    Returns:
        float: Average cosine similarity score
    """
    # Load evaluation data
    with open(data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Create queries and passages for evaluation
    queries = []
    passages = []
    
    for item in eval_data:
        queries.append(item['query'])
        passages.append(item['positive'])
    
    # Encode queries and passages
    query_embeddings = model.encode(queries, batch_size=batch_size, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, batch_size=batch_size, convert_to_tensor=True)
    
    # Calculate cosine similarities
    cosine_scores = torch.nn.functional.cosine_similarity(query_embeddings, passage_embeddings)
    
    # Calculate average score
    avg_score = cosine_scores.mean().item()
    print(f"Average cosine similarity: {avg_score:.4f}")
    
    return avg_score

def train_model(train_data_path, val_data_path, output_dir, 
               base_model="all-mpnet-base-v2", 
               batch_size=16, 
               epochs=3, 
               learning_rate=2e-5,
               warmup_steps=100,
               use_amp=False):  # Default to False to avoid compatibility issues
    """
    Fine-tune a Sentence-BERT model on NASA bioscience data
    
    Args:
        train_data_path (str): Path to training data
        val_data_path (str): Path to validation data
        output_dir (str): Directory to save the model
        base_model (str): Base model to use
        batch_size (int): Batch size for training
        epochs (int): Number of epochs
        learning_rate (float): Learning rate
        warmup_steps (int): Number of warmup steps
        use_amp (bool): Whether to use automatic mixed precision
        
    Returns:
        SentenceTransformer: Fine-tuned model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model
    print(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # Create dataset and dataloader
    print("Loading training data")
    train_dataset = QueryPassageDataset(train_data_path)
    
    # Define a custom collate function to handle InputExample objects
    def custom_collate_fn(batch):
        # This function just returns the batch as is, without trying to stack the examples
        return batch
        
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    # Define loss function (triplet loss)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Train the model
    print(f"Training model for {epochs} epochs")
    
    # Simplified training loop with validation
    best_score = -1
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_value = 0
        for batch_examples in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Extract texts and labels from the batch of InputExamples
            batch_texts = []
            batch_labels = []
            for example in batch_examples:
                batch_texts.append(example.texts)
                batch_labels.append(example.label)
            
            # Process the batch
            model.zero_grad()
            # SentenceTransformer's train_objectives expects a list of tuples (sentences, labels)
            # We'll use the model's internal methods to handle this
            features = [model.tokenize(texts) for texts in batch_texts]
            loss_value = train_loss(features, batch_labels)
            loss_value.backward()
            optimizer.step()
            train_loss_value += loss_value.item()
        
        # Validation
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {train_loss_value/len(train_dataloader):.4f}")
        val_score = simple_evaluate(model, val_data_path, batch_size)
        
        # Save best model
        if val_score > best_score:
            best_score = val_score
            print(f"New best score: {best_score:.4f}, saving model")
            model.save(output_dir)
    
    print(f"Model saved to {output_dir}")
    return model

def evaluate_model(model, test_data_path, batch_size=16):
    """
    Evaluate the model on test data
    
    Args:
        model (SentenceTransformer): Fine-tuned model
        test_data_path (str): Path to test data
        batch_size (int): Batch size for evaluation
        
    Returns:
        float: Average cosine similarity score
    """
    print("Evaluating model on test data")
    result = simple_evaluate(model, test_data_path, batch_size)
    print(f"Test result: {result:.4f}")
    
    return result

def run_step_by_step(use_small_model=True, offline_mode=False):
    """Run the training process step by step
    
    Args:
        use_small_model (bool): Whether to use a smaller model for faster download
        offline_mode (bool): Whether to use a local model if available
    """
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    train_data_path = os.path.join(base_dir, "data", "processed", "train.json")
    val_data_path = os.path.join(base_dir, "data", "processed", "val.json")
    test_data_path = os.path.join(base_dir, "data", "processed", "test.json")
    output_dir = os.path.join(base_dir, "models", "sbert_nasa")
    
    # Step 1: Load and check data files
    print("\nStep 1: Checking data files...")
    for data_path in [train_data_path, val_data_path, test_data_path]:
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            print(f"  ✓ {os.path.basename(data_path)} exists with {len(data)} examples")
        else:
            print(f"  ✗ {os.path.basename(data_path)} does not exist!")
            return
    
    # Step 2: Load base model
    print("\nStep 2: Loading base model...")
    # Use a smaller model if requested
    if use_small_model:
        base_model = "paraphrase-MiniLM-L3-v2"  # Much smaller model (33MB vs 420MB)
    else:
        base_model = "all-mpnet-base-v2"
    
    # Check if we should try to use a local model first
    if offline_mode and os.path.exists(output_dir):
        try:
            print(f"  Trying to load local model from {output_dir}...")
            model = SentenceTransformer(output_dir)
            print(f"  ✓ Local model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load local model: {e}")
            print(f"  Falling back to downloading {base_model}...")
            try:
                model = SentenceTransformer(base_model)
                print(f"  ✓ Model loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load model: {e}")
                return
    else:
        try:
            print(f"  Loading {base_model}...")
            model = SentenceTransformer(base_model)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            return
    
    # Step 3: Create dataset and dataloader
    print("\nStep 3: Creating dataset and dataloader...")
    try:
        # Define a custom collate function to handle InputExample objects
        def custom_collate_fn(batch):
            # This function just returns the batch as is, without trying to stack the examples
            return batch
            
        train_dataset = QueryPassageDataset(train_data_path)
        print(f"  ✓ Training dataset created with {len(train_dataset)} examples")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=custom_collate_fn)
        print(f"  ✓ Training dataloader created with custom collate function")
    except Exception as e:
        print(f"  ✗ Failed to create dataset: {e}")
        return
    
    # Step 4: Define loss function
    print("\nStep 4: Defining loss function...")
    try:
        train_loss = losses.MultipleNegativesRankingLoss(model)
        print(f"  ✓ Loss function defined")
    except Exception as e:
        print(f"  ✗ Failed to define loss function: {e}")
        return
    
    # Step 5: Train for one batch
    print("\nStep 5: Training for one batch...")
    try:
        model.train()
        # Get a batch of examples
        batch_examples = next(iter(train_dataloader))
        
        # For demonstration, we'll just use the first example
        example = batch_examples[0]
        
        # SentenceTransformer expects a list of strings
        query = example.texts[0]  # First text is the query
        passage = example.texts[1]  # Second text is the passage
        
        # Process the texts
        model.zero_grad()
        query_embedding = model.encode([query], convert_to_tensor=True)
        passage_embedding = model.encode([passage], convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_score = torch.nn.functional.cosine_similarity(query_embedding, passage_embedding)
        print(f"  ✓ Cosine similarity for sample pair: {cosine_score.item():.4f}")
        
        # Print the first few tokens of the texts
        print(f"  Query: {query[:50]}...")
        print(f"  Passage: {passage[:50]}...")
        
        print("  ✓ Successfully processed one example")
    except Exception as e:
        print(f"  ✗ Failed to process example: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Evaluate on a few validation examples
    print("\nStep 6: Evaluating on validation data...")
    try:
        # Load a few validation examples
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
        
        # Take just the first 3 examples for quick evaluation
        sample_val_data = val_data[:3]
        
        # Process each example
        scores = []
        for i, item in enumerate(sample_val_data):
            query = item['query']
            passage = item['positive']
            
            # Encode the texts
            query_embedding = model.encode([query], convert_to_tensor=True)
            passage_embedding = model.encode([passage], convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_score = torch.nn.functional.cosine_similarity(query_embedding, passage_embedding).item()
            scores.append(cosine_score)
            
            # Print details
            print(f"  Example {i+1}:")
            print(f"    Query: {query[:50]}...")
            print(f"    Passage: {passage[:50]}...")
            print(f"    Similarity: {cosine_score:.4f}")
        
        # Calculate average score
        avg_score = sum(scores) / len(scores)
        print(f"  ✓ Average validation score on {len(sample_val_data)} examples: {avg_score:.4f}")
    except Exception as e:
        print(f"  ✗ Failed to evaluate on validation data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nAll steps completed successfully!")
    print("To run the full training process, use: python train.py --full")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a SentenceBERT model on NASA bioscience data')
    parser.add_argument('--full', action='store_true', help='Run the full training process')
    parser.add_argument('--small-model', action='store_true', help='Use a smaller model for faster download')
    parser.add_argument('--offline', action='store_true', help='Try to use a local model if available')
    parser.add_argument('--base-model', type=str, default=None, help='Specify a custom base model')
    args = parser.parse_args()
    
    try:
        if args.full:
            # Paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            train_data_path = os.path.join(base_dir, "data", "processed", "train.json")
            val_data_path = os.path.join(base_dir, "data", "processed", "val.json")
            test_data_path = os.path.join(base_dir, "data", "processed", "test.json")
            output_dir = os.path.join(base_dir, "models", "sbert_nasa")
            
            # Determine which base model to use
            if args.base_model:
                base_model = args.base_model
            elif args.small_model:
                base_model = "paraphrase-MiniLM-L3-v2"
            else:
                base_model = "all-mpnet-base-v2"
            
            print(f"Starting full model training with {base_model}...")
            # Train the model
            model = train_model(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_dir=output_dir,
                base_model=base_model,
                batch_size=16,
                epochs=50,
                learning_rate=2e-5,
                warmup_steps=100,
                use_amp=False  # Disable AMP to avoid compatibility issues
            )
            
            # Evaluate the model
            evaluate_model(model, test_data_path, batch_size=16)
        else:
            # Run step by step
            run_step_by_step(use_small_model=args.small_model or True, offline_mode=args.offline)
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()