import os
import requests
import pandas as pd
from tqdm import tqdm

def download_dataset(url, save_path):
    """
    Download the NASA bioscience dataset from GitHub
    
    Args:
        url (str): URL to the raw CSV file on GitHub
        save_path (str): Local path to save the dataset
    """
    print(f"Downloading dataset from {url}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    # Write to file with progress bar
    with open(save_path, 'wb') as f, tqdm(
        desc=save_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
    
    print(f"Dataset saved to {save_path}")
    return save_path

def load_dataset(file_path):
    """
    Load the NASA bioscience dataset
    
    Args:
        file_path (str): Path to the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df

if __name__ == "__main__":
    # URL to the raw CSV file on GitHub
    github_url = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"
    
    # Local path to save the dataset
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "data", "raw", "SB_publication_PMC.csv")
    
    # Download the dataset
    download_dataset(github_url, local_path)
    
    # Load the dataset
    df = load_dataset(local_path)
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Number of publications: {len(df)}")
    print(df.describe())