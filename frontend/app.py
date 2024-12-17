import streamlit as st
from store_vector import VectorStore
from utils import *
import tempfile
import os
import time

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
    min_value=2,
    max_value=2000,
    value=128
)

# Sidebar: Optional user-provided context
st.sidebar.header("Context Configuration")
user_context = st.sidebar.text_area("Optional Context", value="", help="Add a custom context before querying")

# Add a selection for search type
search_type = st.sidebar.radio(
    "Search Type",
    options=["Retrieval-Passage", "Question-Answering"],
    index=1  # Default to Question-Answering
)

# Initialize vector store
if 'vector_store' not in st.session_state:
    sample_chunk = ["This is a sample text."]
    embedding_dim = len(generate_embeddings(sample_chunk)[0])
    st.session_state.vector_store = VectorStore(dim=embedding_dim)

# Initialize a set to track processed PDFs
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = set()

# Modified to accept multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Process PDFs button
if uploaded_files: #and st.button("Process PDFs"):
    chunking_config = ChunkingConfig(
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=0.15
    )

    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)

        # Check if already processed
        if file_hash not in st.session_state.processed_pdfs:
            # Check for saved embeddings
            embeddings, chunks = load_embeddings(file_hash)

            if embeddings and chunks:
                st.info(f"Loaded cached embeddings for '{uploaded_file.name}'")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    pdf_path = tmp_file.name

                # Extract text and process chunks
                text = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(text, chunking_config)
                embeddings = generate_embeddings(chunks, task="retrieval.passage")

                # Save embeddings for future use
                save_embeddings(file_hash, embeddings, chunks)

                # Clean up the temporary file
                os.unlink(pdf_path)

            # Add embeddings and chunks to the vector store
            st.session_state.vector_store.add_vectors(embeddings, chunks)

            # Mark the file as processed
            st.session_state.processed_pdfs.add(file_hash)
    
    st.success(f"Processed {len(st.session_state.processed_pdfs)} unique PDFs. Total chunks: {st.session_state.vector_store.get_total_chunks()}")


# Search interface
query = st.text_input("Enter your search query")

if query and st.session_state.vector_store:
    # Record start time
    start_time = time.time()

    #Generate Query-Embedding
    query_embedding = generate_embeddings([query], task="retrieval.passage")[0]
    similar_chunks, distances = st.session_state.vector_store.query(query_embedding, top_k=5)
    
    # Combine the top-k chunks into a single context
    context = " ".join(similar_chunks)
    if user_context:
        context = user_context + " " + context  # Append user-provided context
    
    # Display time taken
    retrieval_time = time.time() - start_time
    
    if search_type == "Retrieval-Passage":
        st.subheader("Search Results")
        st.info(f"Time taken to retrieve results: {retrieval_time:.2f} seconds")
        
        for chunk, distance in zip(similar_chunks, distances):
            st.markdown(f"**Similarity Score: {1 / (1 + distance):.3f}**")
            st.text(chunk)
            st.markdown("---")

    elif search_type == "Question-Answering":
        # Use a generative model to answer the query based on context
        raw_answer = generate_answer(query, context)  # New function to generate answer
        answer = summarize_answer(similar_chunks, query)
        

        st.subheader("Answer")
        st.info(f"Time taken to retrieve results: {retrieval_time:.2f} seconds")
        st.markdown(f"**{answer}**")
        
        st.subheader("Supporting Context")
        for chunk, distance in zip(similar_chunks, distances):
            st.markdown(f"**Similarity Score: {1 / (1 + distance):.3f}**")
            st.text(chunk)
            st.markdown("---")
