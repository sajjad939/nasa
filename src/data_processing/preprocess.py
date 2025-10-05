import os
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Clean and normalize text data
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and normalize whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    return text.strip()

def segment_document(text, max_length=512):
    """
    Segment document into chunks of specified maximum length
    
    Args:
        text (str): Document text
        max_length (int): Maximum chunk length
        
    Returns:
        list: List of text segments
    """
    if pd.isna(text) or not isinstance(text, str) or len(text) == 0:
        return []
    
    # Simple segmentation by sentences and then combining into chunks
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk)
    
    return chunks

def create_query_passage_pairs(df):
    """
    Create query-passage pairs for training
    
    Args:
        df (pd.DataFrame): Input dataframe with text data
        
    Returns:
        list: List of dictionaries with query-passage pairs
    """
    pairs = []
    
    # Use title as query and generate synthetic passages from titles
    # since we don't have abstracts in this dataset
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating query-passage pairs"):
        title = clean_text(row.get('Title', ''))
        link = row.get('Link', '')
        
        if len(title) < 10:  # Skip entries with insufficient data
            continue
        
        # Use the title as both query and passage for demonstration purposes
        # In a real scenario, you would fetch abstracts from the links
        pairs.append({
            'query': title,
            'positive': title,  # Using title as passage for demonstration
            'link': link,
            'segment_id': 0
        })
        
        # Create some variations for more training data
        words = title.split()
        if len(words) > 5:
            # Create a variation by taking first half of title
            half_point = len(words) // 2
            variation = ' '.join(words[:half_point])
            
            pairs.append({
                'query': variation,
                'positive': title,
                'link': link,
                'segment_id': 1
            })
    
    return pairs

def preprocess_dataset(input_path, output_dir):
    """
    Preprocess the NASA bioscience dataset
    
    Args:
        input_path (str): Path to the raw dataset
        output_dir (str): Directory to save processed data
        
    Returns:
        dict: Paths to the processed datasets
    """
    print(f"Preprocessing dataset from {input_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic cleaning
    print("Cleaning text data...")
    for col in ['Title']:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Create query-passage pairs
    print("Creating query-passage pairs...")
    pairs = create_query_passage_pairs(df)
    print(f"Created {len(pairs)} query-passage pairs")
    
    # Check if we have pairs before splitting
    if len(pairs) == 0:
        print("Error: No query-passage pairs were created. Check your dataset.")
        return {}
    
    # Split into train, validation, and test sets
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1, random_state=42)
    
    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")
    print(f"Test set: {len(test_pairs)} pairs")
    
    # Save processed datasets
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_pairs, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_pairs, f)
    
    with open(test_path, 'w') as f:
        json.dump(test_pairs, f)
    
    print(f"Processed datasets saved to {output_dir}")
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_path = os.path.join(base_dir, "data", "raw", "SB_publication_PMC.csv")
    output_dir = os.path.join(base_dir, "data", "processed")
    
    # Preprocess the dataset
    preprocess_dataset(input_path, output_dir)