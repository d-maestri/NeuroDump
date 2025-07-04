from processing import *
from config import *


# ### MAIN PROCESSING PIPELINE
def process_file(fname):
    """Process a single file through the complete pipeline."""
    print(f"\n\n=============== PROCESSING FILE: {fname} ===============")

    # Step 1: Read file
    with open(os.path.join(FOLDER_PATH, fname), "r", encoding="utf-8") as f:
        text = f.read()

    # Step 2: Create chunks
    chunks, metadata = chunk_text(text, fname, inspect=True)        # Split text into chunks
    print(f"\n=== Created {len(chunks)} chunks")

    # Step 3: Store in vector database
    vector_manager = VectorStoreManager()    # Initialize vector store
    vector_manager.add_chunks(chunks, metadata)                     # Add chunks with embeddings
    print(f"\n=== Stored {len(chunks)} chunks in Chroma DB")
    
    # Step 3.1: Mark as processed
    with open(LOG_PATH, "a") as log:                                # Log processed file
        log.write(fname + "\n")

    # Step 4: Cluster chunks
    print(f"\n=== CLUSTERING CHUNKS FROM {fname}")
    chunks, embeddings, metadatas = vector_manager.get_chunks_with_embeddings(fname) # Retrieve stored data

    clusterizer = ChunkClusterizer(chunks, embeddings, metadatas, fname)             # Initialize clustering
    clusterizer.cluster_chunks()                                            # Perform clustering
    clusterizer.save_clusters()                                             # Save clusters

    # STEP 5: Read clustered JSON file
    clusters_path = os.path.join(CLUSTER_DIR, fname.replace(".txt", "_clusters.json"))
    note_clusters = read_note_clusters(clusters_path)

    # STEP 6: Query LLM with SYSTEM_PROMPT_RECLUSTER
    model = "mistral"
    prompt = SYSTEM_PROMPT_RECLUSTER + f"\n\nHERE IS THE CLUSTER DATA FROM {fname}:\n" + note_clusters + "\n\nPLEASE PERFORM THE RECLUSTERING AND CLUSTER NAMING TASKS."
    response = query_ollama(model, prompt)

    # STEP 6.1: Display reasoning(Markdown)
    reasoning_part = response.split("CLUSTER_NAMES:")[0]
    display(Markdown(reasoning_part.strip()))
    print(response)

    # STEP 7: Parse and save JSON outputs
    cluster_names, note_reclustered = parse_response(response)
    print(cluster_names)
    print(note_reclustered)

    # STEP 7.1: Save 
    reclustered_path, note_reclustered, CLUSTER_NAMES_DICT_PATH  = save_to_files(cluster_names, note_reclustered, fname)

    # STEP 8: Split note reclustered into formatted subnotes
    reclustered_filename = os.path.basename(reclustered_path)
    split_clusters_to_formatted_subnotes(fname_RECLUSTERED=reclustered_filename)
    return response


def process_all_new():
    """Process all new files in the input directory."""
    new_files = get_new_files()                                     # Get list of unprocessed files
    
    if not new_files:                                               # Exit early if no new files
        print("\n=== No new files to process.")
        return
    
    print(f"\n=== Found {len(new_files)} new files: {new_files}")
    
    for fname in new_files:                                         # Process each new file
        process_file(fname)
    
    print(f"\n=== Completed processing {len(new_files)} files")