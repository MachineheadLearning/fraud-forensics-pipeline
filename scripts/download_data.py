import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    # Define dataset and target path
    dataset_id = "kartik2112/fraud-detection"
    target_dir = "data/raw"
    
    # Initialize and Authenticate
    # This automatically looks for ~/.kaggle/kaggle.json
    api = KaggleApi()
    api.authenticate()
    
    # Download files (unzip=True handles the extraction for you)
    print(f"Downloading {dataset_id}...")
    api.dataset_download_files(dataset_id, path=target_dir, unzip=True)
    print("Files downloaded and unzipped successfully.")

if __name__ == "__main__":
    download_dataset()