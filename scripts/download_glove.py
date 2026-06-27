#!/usr/bin/env python3
"""Safely download and extract GloVe embeddings."""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.config import DATA_DIR, GLOVE_PATH  # noqa: E402

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
ZIP_PATH = DATA_DIR / "glove.6B.zip"

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading GloVe: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
    sys.stdout.flush()

def main():
    if GLOVE_PATH.exists():
        print(f"GloVe file already exists at {GLOVE_PATH}. Skipping download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("WARNING: This will download an ~822MB file from Stanford University.")
    print("It may take a few minutes depending on your internet speed.")
    
    try:
        if not ZIP_PATH.exists():
            urllib.request.urlretrieve(GLOVE_URL, ZIP_PATH, download_progress)
            print("\nDownload complete!")
        
        print(f"Extracting {GLOVE_PATH.name}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Only extract the 300d version to save space
            zip_ref.extract(GLOVE_PATH.name, path=DATA_DIR)
            
        print("Extraction complete!")
        
        # Cleanup zip
        print("Cleaning up zip file to save space...")
        os.remove(ZIP_PATH)
        
        print("GloVe is ready to use!")
        
    except Exception as e:
        print(f"\nError downloading or extracting GloVe: {e}")
        if ZIP_PATH.exists():
            os.remove(ZIP_PATH)
        sys.exit(1)

if __name__ == "__main__":
    main()
