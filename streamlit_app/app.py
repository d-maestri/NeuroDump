import streamlit as st
import os
import json
from pathlib import Path
from pipeline import process_file  # Your existing function
from processing import extract_base_filename

# === Directories ===
DATA_DIR = Path("./data")
CLUSTER_DIR = DATA_DIR / "cluster_dir"
LLM_RECLUSTER_DIR = DATA_DIR / "outputs" / "LLM_recluster"
SUBNOTES_DIR = DATA_DIR / "outputs" / "SUBNOTES"

st.set_page_config(page_title="NeuroNote Processor", layout="wide")

st.title("🧠 NeuroDump Processor")
st.write("Organize your messy notes into structured knowledge with clustering and LLM magic.")

# === Initialize Session State ===
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'selected_base_fname' not in st.session_state:
    st.session_state.selected_base_fname = None

# === Helper Function to Discover Processed Files ===
def get_processed_files():
    """Scan folders to find all previously processed files"""
    processed_files = {}
    
    # Look for cluster files in CLUSTER_DIR
    if CLUSTER_DIR.exists():
        for cluster_file in CLUSTER_DIR.glob("*_clusters.json"):
            base_name = cluster_file.stem.replace("_clusters", "")
            processed_files[base_name] = {
                'base_fname': base_name,
                'cluster_json_path': cluster_file,
                'recluster_json_path': LLM_RECLUSTER_DIR / f"{base_name}_RECLUSTERED.json",
                'cluster_names_path': LLM_RECLUSTER_DIR / "CLUSTER_NAMES_DICT.json",
                'cluster_file_mtime': cluster_file.stat().st_mtime
            }
    
    return processed_files

# === Input Section ===
note_input = st.text_area("Paste your note here", height=200)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

run_pipeline = st.button("🚀 Run Note Pipeline")

if run_pipeline:
    if not note_input and not uploaded_file:
        st.warning("Please paste a note or upload a .txt file.")
    else:
        # === Save Input ===
        if uploaded_file:
            note_content = uploaded_file.read().decode("utf-8")
            fname = uploaded_file.name
        else:
            note_content = note_input
            fname = "user_note.txt"

        input_path = DATA_DIR / fname
        input_path.write_text(note_content, encoding="utf-8")

        # === Process File ===
        st.info("Processing your note...")
        response = process_file(fname)
        st.success("✅ Done!")

        # === Store Results in Session State ===
        base_fname = extract_base_filename(fname)
        st.session_state.processed_data = {
            'base_fname': base_fname,
            'original_filename': fname,
            'response': response,
            'cluster_json_path': CLUSTER_DIR / f"{base_fname}_clusters.json",
            'recluster_json_path': LLM_RECLUSTER_DIR / f"{base_fname}_RECLUSTERED.json",
            'cluster_names_path': LLM_RECLUSTER_DIR / "CLUSTER_NAMES_DICT.json",
            'just_processed': True
        }
        st.session_state.selected_base_fname = base_fname
        st.session_state.processing_complete = True

# === Previously Processed Files Selector ===
processed_files = get_processed_files()

if processed_files:
    st.subheader("📋 Previously Processed Files")
    
    # Create display options for selectbox
    file_options = []
    for base_name, file_info in processed_files.items():
        # Get file modification time for display
        import datetime
        mtime = datetime.datetime.fromtimestamp(file_info['cluster_file_mtime'])
        display_name = f"{base_name} (processed: {mtime.strftime('%Y-%m-%d %H:%M')})"
        file_options.append((display_name, base_name))
    
    # Sort by modification time (newest first)
    file_options.sort(key=lambda x: processed_files[x[1]]['cluster_file_mtime'], reverse=True)
    
    # Add "None" option at the beginning
    display_options = ["-- Select a processed file --"] + [option[0] for option in file_options]
    
    selected_display = st.selectbox(
        "Select a previously processed file to view/download:",
        display_options,
        index=0
    )
    
    # Update selection based on dropdown
    if selected_display != "-- Select a processed file --":
        selected_base_fname = next(base_name for display, base_name in file_options if display == selected_display)
        
        # Load the selected file data
        selected_file_info = processed_files[selected_base_fname]
        
        # Try to load the response from recluster file if it exists
        response = None
        if selected_file_info['recluster_json_path'].exists():
            try:
                with open(selected_file_info['recluster_json_path'], 'r', encoding='utf-8') as f:
                    recluster_data = json.load(f)
                    response = recluster_data.get('llm_response', '')
            except:
                response = None
        
        st.session_state.processed_data = {
            'base_fname': selected_base_fname,
            'original_filename': selected_base_fname,  # We don't know the original filename
            'response': response,
            'cluster_json_path': selected_file_info['cluster_json_path'],
            'recluster_json_path': selected_file_info['recluster_json_path'],
            'cluster_names_path': selected_file_info['cluster_names_path'],
            'just_processed': False
        }
        st.session_state.selected_base_fname = selected_base_fname
        st.session_state.processing_complete = True

# === Display Results (only if processing is complete) ===
if st.session_state.processing_complete and st.session_state.processed_data:
    data = st.session_state.processed_data
    
    # Show which file is currently displayed
    if data.get('just_processed', False):
        st.info(f"📄 Currently displaying: **{data.get('original_filename', 'Unknown')}** (just processed)")
    else:
        st.info(f"📄 Currently displaying: **{data.get('base_fname', 'Unknown')}** (from files)")
    
    # === Display Cluster Distribution ===
    st.subheader("📊 Cluster Distribution")
    if data['cluster_json_path'].exists():
        with open(data['cluster_json_path'], 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        for cid, chunks in clusters.items():
            label = "Noise" if cid == "-1" else f"Cluster {cid}"
            st.write(f"**{label}**: {len(chunks)} chunks")
    else:
        st.warning("No cluster file found.")

    # === Display Reasoning ===
    if data['response'] and "CLUSTER_NAMES:" in data['response']:
        reasoning_part = data['response'].split("CLUSTER_NAMES:")[0].strip()
        st.subheader("🧠 LLM Reasoning")
        
        # Use a container with custom CSS for text wrapping
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #ff6b6b;
                font-family: 'Courier New', monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
                overflow-x: auto;
            ">
            {reasoning_part}
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.warning("No reasoning found in the LLM response.")


    # === Download Cluster Files ===
    st.subheader("⬇️Download JSON Files")

    def offer_download(path, label, key_suffix):
        if path.exists():
            with open(path, "rb") as f:
                st.download_button(
                    label=label, 
                    data=f, 
                    file_name=path.name, 
                    mime="application/json",
                    key=f"download_{key_suffix}"
                )

    offer_download(data['cluster_json_path'], "⬇️ Download Clusters JSON", "clusters")
    offer_download(data['recluster_json_path'], "⬇️ Download Reclustering JSON", "recluster")
    offer_download(data['cluster_names_path'], "⬇️ Download Cluster Names Dictionary", "cluster_names")

    # === Download SUBNOTES ===
    st.subheader("⬇️Download Cluster Subnotes")
    
    # Debug info (you can remove this later)
    st.write(f"Looking for subnotes with pattern: {data['base_fname']}_*.txt")
    
    subnote_files = list(SUBNOTES_DIR.glob(f"{data['base_fname']}_*.txt"))
    if not subnote_files:
        st.info("No subnote files found. Check if the processing generated subnotes.")
    else:
        for i, file in enumerate(sorted(subnote_files)):
            with open(file, "rb") as f:
                st.download_button(
                    label=f"⬇️ {file.name}",
                    data=f,
                    file_name=file.name,
                    mime="text/plain",
                    key=f"subnote_{i}"
                )

    # === Reset Button ===
    if st.button("🔄 Process New Note"):
        st.session_state.processing_complete = False
        st.session_state.processed_data = {}
        st.session_state.selected_base_fname = None
        st.rerun()