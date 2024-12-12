import streamlit as st
from src.store_vector import VectorStore
from src.utils import extract_text_from_pdf, chunk_text, ChunkingConfig, generate_embeddings
import tempfile
from sentence_transformers import SentenceTransformer

st.title("PDF Search with Embeddings")

# Sidebar for chunking configuration
st.sidebar.header("Chunking Configuration")
chunk_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    ["paragraph", "word", "character"],
    index=1  # Default to word
)
chunk_size = st.sidebar.number_input(
    "Chunk Size",
    min_value=100,
    max_value=2000,
    value=128
)

# Initialize vector store
if 'vector_store' not in st.session_state:
    sample_chunk = ["This is a sample text."]
    embedding_dim = len(generate_embeddings(sample_chunk)[0])
    st.session_state.vector_store = VectorStore(dim=embedding_dim)

# Modified to accept multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Process PDFs button
if uploaded_files and st.button("Process PDFs"):
    chunking_config = ChunkingConfig(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=None
    )
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, chunking_config)
        embeddings = generate_embeddings(chunks, task="retrieval.passage")
        st.session_state.vector_store.add_vectors(embeddings, chunks)
    
    st.success(f"Processed {len(uploaded_files)} PDFs. Total chunks: {st.session_state.vector_store.get_total_chunks()}")

# Search interface
query = st.text_input("Enter your search query")

if query and st.session_state.vector_store:
    query_embedding = generate_embeddings([query], task="retrieval.passage")[0]
    similar_chunks, distances = st.session_state.vector_store.query(query_embedding)
    
    st.subheader("Search Results")
    for chunk, distance in zip(similar_chunks, distances):
        st.markdown(f"**Similarity Score: {1 / (1 + distance):.3f}**")
        st.text(chunk)
        st.markdown("---")