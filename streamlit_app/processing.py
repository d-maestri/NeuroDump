#imports
import os
import re
import nltk
from transformers import AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from IPython.display import Markdown, display
import hdbscan
import numpy as np
import json
from collections import defaultdict
import subprocess
import ast

nltk.download('punkt')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import *


# Initialize embedding model
HF_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Generic Save JSON file 
def save_json(data, folder_path, filename):
    """Save data as JSON to specified folder and filename."""
    out_path = os.path.join(folder_path, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n=== Saved JSON to: {out_path}")
    return out_path


def extract_base_filename(filename):
    """Extract base filename without extension and suffixes."""
    # Remove .txt extension if present
    if filename.endswith('.txt'):
        filename = filename[:-4]
    
    #Remove _RECLUSTERED.json if present
    if filename.endswith('_RECLUSTERED.json'):
        filename = filename[:-17]

    # Remove _clusters suffix if present
    if filename.endswith('_clusters'):
        filename = filename[:-9]
    
    return filename


def get_processed_files():
    """Load list of already processed files from log."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return set(line.strip() for line in f)
    return set()


def get_new_files():
    """Get list of new .txt files that haven't been processed yet."""
    if not os.path.exists(FOLDER_PATH):
        return []
    
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".txt")]
    processed_files = get_processed_files()
    return [f for f in all_files if f not in processed_files]


### 2. Split document into chunks of tokens
def count_tokens(text):
    """Count tokens in text using the embedding model's tokenizer."""
    return len(tokenizer.tokenize(text))

def is_code_like(sentence):
    """Detect if a sentence contains code-like patterns."""
    code_keywords = ["from ", "import ", "def ", "return ", "class ", "for ", "if ", "else", "while ", "try:", "except", "print("]
    code_symbols = ["=", "()", "[]", "{}", "->", "::"]
    sentence_lower = sentence.strip().lower()
    return (
        any(sentence_lower.startswith(k) for k in code_keywords) or
        any(sym in sentence for sym in code_symbols)
    )

# ### CHUNKING LOGIC
def chunk_text(text, source_name, min_tokens=40, max_tokens=100, inspect=False):
    """Split text into semantic chunks with token limits."""
    raw_blocks = re.split(r"\n\s*\n", text.strip())                    # Split by double newlines
    chunks, current_chunk, current_tokens = [], [], 0                  # Initialize chunking variables
    metadata = []                                                      # Store metadata for each chunk

    for block in raw_blocks:                                           # Process each text block
        sentences = nltk.sent_tokenize(block.strip())                  # Split block into sentences

        for sentence in sentences:                                     # Process each sentence
            sentence = sentence.strip()
            tokens = count_tokens(sentence)                            # Count tokens in sentence

            # Handle code-like sentences separately
            if is_code_like(sentence):                                 # Check if sentence contains code
                if current_chunk:                                      # Save current chunk if exists
                    chunks.append(" ".join(current_chunk))
                    metadata.append({"source": source_name})
                    current_chunk, current_tokens = [], 0
                chunks.append(sentence)                                # Add code as separate chunk
                metadata.append({"source": source_name})
                continue

            # Handle conversation transitions
            if any(sentence.lower().startswith(w) for w in ["btw", "wait", "oh", "then", "also", "now", "next"]):
                if current_chunk:                                      # Break chunk at transition words
                    chunks.append(" ".join(current_chunk))
                    metadata.append({"source": source_name})
                    current_chunk, current_tokens = [], 0

            # Check if adding sentence would exceed max tokens
            if current_tokens + tokens > max_tokens and current_tokens >= min_tokens:
                chunks.append(" ".join(current_chunk))                # Save current chunk
                metadata.append({"source": source_name})
                current_chunk, current_tokens = [], 0                 # Reset for new chunk

            current_chunk.append(sentence)                            # Add sentence to current chunk
            current_tokens += tokens                                  # Update token count

        # Save any remaining chunk after block processing
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            metadata.append({"source": source_name})
            current_chunk, current_tokens = [], 0

    # if inspect:                                                       # Optional: print chunks for debugging
    #     for i, chunk in enumerate(chunks):
    #         print(f"\n=== Chunks ===")
    #         print(f"\nChunk {i+1} ({count_tokens(chunk)} tokens):\n{chunk}")

    return chunks, metadata


### 3. Embeddings and save vectors in Chroma DB
# ### VECTOR STORE MANAGEMENT
class VectorStoreManager:
    def __init__(self):
        self.persist_dir = PERSIST_DIR                              # Store database directory path
        self.embeddings = HF_embeddings                          # Store embedding model
        self.vectorstore = self._load_existing()                    # Load existing DB or set to None

    def _load_existing(self):
        """Load existing Chroma database if it exists."""
        try:
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                print(f"\n=== Loading existing Chroma DB from {self.persist_dir}")
                return Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="default"
                )
        except Exception as e:
            print(f"\n=== Error loading existing Chroma DB: {e}")
        
        print("\n=== No existing Chroma DB found. Will create new one.")
        return None

    def add_chunks(self, chunks, metadata):
        """Add text chunks to vector store with embeddings."""
        if not self.vectorstore:                                     # Create new database if none exists
            print(f"\n=== Creating new Chroma DB at {self.persist_dir}")
            self.vectorstore = Chroma.from_texts(
                texts=chunks,
                metadatas=metadata,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="default"
            )
        else:                                                        # Add to existing database
            self.vectorstore.add_texts(texts=chunks, metadatas=metadata)


    def get_chunks_with_embeddings(self, fname):
        """Retrieve chunks, embeddings, and metadata for a specific file."""
        results = self.vectorstore.get(
            where={"source": fname},
            include=["documents", "embeddings", "metadatas"]
        )
        chunks = results.get('documents', [])               # Extract text chunks
        embeddings_list = results.get('embeddings', [])     # Exttact embeddings
        metadatas = results.get('metadatas', [])            # Extract metadata
    
        return chunks, embeddings_list, metadatas        
    

# ### CLUSTERING LOGIC
class ChunkClusterizer:
    def __init__(self, chunks, chunk_embeddings, metadata, fname):
        self.chunks = chunks                                         # Store text chunks
        self.embeddings = np.array(chunk_embeddings)                 # Convert embeddings to numpy array for ML operations
        self.metadata = metadata                                     # Save metadata       
        self.fname = fname
        self.labels = []                                             # Initialize cluster labels
        self.clusters = defaultdict(list)                            # cluster_id -> list of dicts {chunk, metadata}


    def cluster_chunks(self, min_cluster_size=2, min_samples=1):
        """Cluster chunks using HDBSCAN algorithm."""
        
        # Handle edge cases
        if len(self.chunks) < min_cluster_size:                      # Not enough chunks to cluster
            print(f"\n=== Not enough chunks to cluster ({len(self.chunks)} < {min_cluster_size}) ===")
            self.labels = [-1] * len(self.chunks)                   # Assign all to noise
            return self.labels

        # Ensure embeddings are 2D
        if self.embeddings.ndim == 1:                               # Reshape 1D embeddings to 2D
            self.embeddings = self.embeddings.reshape(1, -1)
            
        #print(f"\n===> Clustering {len(self.chunks)} chunks with embeddings shape: {self.embeddings.shape}")
            

        # Perform clustering
        clusterizer = hdbscan.HDBSCAN(                                     # Initialize HDBSCAN clustering algorithm
            min_cluster_size=min_cluster_size,                             # Minimum number of points in a cluster
            min_samples=min_samples,                                       # Minimum samples in neighborhood for core point
            metric='euclidean'                                             # Distance metric for clustering
        )
        self.labels = clusterizer.fit_predict(self.embeddings)             # Fit model and predict cluster labels = generate cluster labels

        # Display results
        num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)  # Count clusters (exclude noise)
        print(f"\n=== Found {num_clusters} clusters")
        print("\n=== Cluster distribution ===")
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            cname = "Noise" if label == -1 else f"Cluster {label}"
            print(f"{cname}: {count} chunks")    
                    
        # Show cluster contents
        for cluster_id in sorted(set(self.labels)):
            cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            print(f"\n=== {cluster_name} ===")

            for i, label in enumerate(self.labels):
                if label == cluster_id:
                    # Save the chunk into the cluster dictionary with global ID
                    self.clusters[str(cluster_id)].append({
                        "chunk_id": i,                      # Global ID
                        "chunk": self.chunks[i],
                        "metadata": self.metadata[i]
                    })

                    # Display clearly: Chunk [global id]
                    display(Markdown(f"**Chunk {i}:** {self.chunks[i]}"))
            
        return self.labels                                   # Return cluster labels
        
    def save_clusters(self):
        filename = self.fname.replace(".txt", "_clusters.json")
        return save_json(self.clusters, CLUSTER_DIR, filename)
    

### 5. Read notes clustered by HDBSCAN
# 5. Read notes clustered by HDBSCAN (Updated to work with any cluster file)
def read_note_clusters(clusters_file_path):
    """Read HDBSCAN clusters file from any path"""
    with open(clusters_file_path, "r", encoding="utf-8") as f:
        note_clusters = f.read()
    return note_clusters


### 6. Define system prompts and initiate LLM
#### 6.1. Prompt: recluster and generate cluster names
SYSTEM_PROMPT_RECLUSTER = """
You are a smart assistant helping organize fragmented personal notes.

The notes were previously split into chunks and clustered semantically. Each original note has its own clusters, saved as JSON files inside: ./data/cluster_dir/. Your task is to refine those clusters and give them a name.

1. REORGANIZE CLUSTERS PER FILE:
- You may SPLIT or MERGE clusters WITHIN EACH FILE to better reflect topic boundaries.
- Keep a balance when creating the new clusters. The topic should NOT be too granular (e.g., splitting KMeans into 3 separate clusters), but also not too broad (e.g., grouping all machine learning together).
- The goal is to create meaningful, reusable sub-notes by topic. Each cluster should contain all content from the original note that belongs to that topic.
- For example, all KMeans content from a note should be together, not scattered across separate clusters.
- Imagine someone searching "KMeans": everything relevant from the note should be in a single rewritten cluster.

2. CLUSTER REORGANIZATION OUTPUT:
2.1. REASONING:
- provide a short reasoning section explaining what you split, merged, or reassigned and why. Format it as normal text.
2.2. 
- Then output A CLEAR JSON OBJECT, with label exactly like this:

NOTE_RECLUSTERED:
{ ... }

2.3.
- Then output a SECOND JSON OBJECT, with label exactly like this:

CLUSTER_NAMES:
{ ... }


- NOTE_RECLUSTERED contains the new clustering structure, following the same format as the input cluster JSON.

- CLUSTER_NAMES is a dictionary mapping new cluster IDs created during reclustering (e.g., "cluster_0") to topic names.

3. FORMATTING RULES (STRICT):
- Keep the reasoning at the top.
- After reasoning, print CLUSTER_NAMES: followed by the RAW JSON object (NO code blocks, NO quotes, NO markdown).
- Then print NOTE_RECLUSTERED: followed by the RAW JSON (same rules).
- DO NOT use triple backticks, ```json, or any Markdown formatting. Just the plain, raw JSON object after the tag.
- Make sure CLUSTER_NAMES and NOTE_RECLUSTERED appear exactly like shown. They are anchors for parsing.
- CLUSTER_NAMES topic names must be FOLDER-SAFE (use underscores, no spaces or special chars).
DO NOT use placeholder text like '...' inside JSON.
DO NOT include comments like `// ...` inside JSON.
Make sure all JSON is valid and can be parsed with json.loads().
- Example:

CLUSTER_NAMES:
{ "cluster_0": "unsupervised_learning", ... }

NOTE_RECLUSTERED:
{ "cluster_0": [ ... ], ... }

You will receive a single note cluster JSON. Analyze and return the result using the format above.
"""


#### 6.2. Initiate LLM
# Initiate Ollama
def query_ollama(model: str, prompt: str) -> str:
    """Initiate Ollama"""
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=prompt)
    if stderr:
        print("OLLAMA STDERR:", stderr)
    return stdout


### 7. Extract response and save CLUSTER_NAMES and NOTE_x_RECLUSTERED
CLUSTER_NAMES_DICT_PATH = os.path.join(LLM_RECLUSTER, "CLUSTER_NAMES_DICT.json")


def parse_response(response_text):
    # Extract CLUSTER_NAMES section
    cluster_names_match = re.search(r'CLUSTER_NAMES:\s*(\{[^}]*\})', response_text, re.DOTALL)
    if cluster_names_match:
        cluster_names_str = cluster_names_match.group(1)
        try:
            cluster_names = json.loads(cluster_names_str)
        except json.JSONDecodeError:
            # Handle potential formatting issues
            cluster_names = eval(cluster_names_str)
    else:
        cluster_names = {}
    
    # Extract NOTE_RECLUSTERED section
    note_reclustered_section = response_text.split("NOTE_RECLUSTERED:", 1)[-1].strip()
    start = note_reclustered_section.find("{")
    if start != -1:
        json_str = note_reclustered_section[start:]
        try:
            note_reclustered = json.loads(json_str)
        except json.JSONDecodeError:
            note_reclustered = parse_complex_json(json_str)
    else:
        note_reclustered = {}
    
    return cluster_names, note_reclustered

    
    
# ==== Save NOTE_RECLUSTERED ====
def save_to_files(cluster_names, note_reclustered, fname, output_dir=LLM_RECLUSTER):
    """
    Save reclustered note with original name and update global CLUSTER_NAMES_DICT.
    """

    base_fname = extract_base_filename(fname)

    # Save NOTE_RECLUSTERED
    reclustered_path = os.path.join(output_dir, f"{base_fname}_RECLUSTERED.json")
    with open(reclustered_path, 'w', encoding='utf-8') as f:
        json.dump(note_reclustered, f, indent=2, ensure_ascii=False)

    # Load or create CLUSTER_NAMES_DICT
    if os.path.exists(CLUSTER_NAMES_DICT_PATH):
        with open(CLUSTER_NAMES_DICT_PATH, 'r', encoding='utf-8') as f:
            cluster_names_dict = json.load(f)
    else:
        cluster_names_dict = {}

    # Update dictionary with new clusters (overwrite if same key)
    cluster_names_dict.update(cluster_names)

    # Save updated CLUSTER_NAMES_DICT
    with open(CLUSTER_NAMES_DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(cluster_names_dict, f, indent=2, ensure_ascii=False)

    print("Files saved:")
    print(f"  - {reclustered_path}")
    print(f"  - {CLUSTER_NAMES_DICT_PATH}")

    return reclustered_path, note_reclustered, CLUSTER_NAMES_DICT_PATH



def parse_complex_json(json_str):
    """Safer fallback parser."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_str)
        except:
            return {}
        

### 8. Split reclustered note into formatted subnotes
def split_clusters_to_formatted_subnotes(fname_RECLUSTERED, cluster_names_dict_path=CLUSTER_NAMES_DICT, output_folder=SUBNOTES):
    """Split reclustered JSON file by cluster and save each as a formatted .txt file using cluster name."""

    base_fname = extract_base_filename(fname_RECLUSTERED)
    reclustered_note_path = os.path.join(LLM_RECLUSTER, fname_RECLUSTERED)

    # === Load reclustered note content ===
    with open(reclustered_note_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # === Load cluster names dict ===
    if isinstance(cluster_names_dict_path, str):
        with open(cluster_names_dict_path, 'r', encoding='utf-8') as f:
            cluster_names_dict = json.load(f)
    else:
        cluster_names_dict = cluster_names_dict_path  # in case it's already a dict

    formatted_subnotes = []

    for cluster_id, chunks in data.items():
        cluster_name = cluster_names_dict.get(cluster_id, cluster_id)
        output_filename = f"{base_fname}_{cluster_name}.txt"
        output_file = os.path.join(output_folder, output_filename)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f"Topic: {cluster_name}\n")
            f_out.write("=" * 50 + "\n\n")
            for chunk_data in chunks:
                chunk_text = chunk_data.get("chunk", "").strip()
                if chunk_text:
                    f_out.write(f"• {chunk_text}\n")
            f_out.write(f"\n[Total: {len(chunks)} notes]\n")

        formatted_subnotes.append(output_file)
        print(f"Created {output_file} with {len(chunks)} chunks")

    print(f"\nAll formatted cluster files saved to '{output_folder}'")
    return formatted_subnotes