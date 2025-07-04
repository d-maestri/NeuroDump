# config.py
from pathlib import Path
import os

# Config paths
FOLDER_PATH = "./data/mock_notes/"          #files
LOG_PATH = "./data/processed_files.txt"     #list of processed files
PERSIST_DIR = "./data/chroma_db"            #embeddings database (store vectorized chunks)
CLUSTER_DIR = "./data/cluster_dir/"         #clusters database
SUBNOTES = "./data/outputs/SUBNOTES"        #subnotes formatted (chunks list)
LLM_RECLUSTER = "./data/outputs/LLM_recluster"
CLUSTER_NAMES_DICT = "./data/outputs/LLM_recluster/cluster_names_dict.json"
CLUSTER_NAMES_DICT_PATH = "./data/outputs/LLM_recluster/cluster_names_dict.json"

# Create directories if they don't exist
os.makedirs(FOLDER_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(SUBNOTES, exist_ok=True)
os.makedirs(LLM_RECLUSTER, exist_ok=True)

