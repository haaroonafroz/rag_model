import streamlit as st
import time
from rag_core import (
    ensure_folders_exist,
    initialize_retriever,
    create_chain,
    setup_chain,
    load_chat_history,
    save_chat_history,
    clear_chat_history,
    update_retriever_k,
    PDF_FILE,
    MODEL_NAME,  # Changed from COLBERT_MODEL
    GENERATOR_MODEL,
    DEFAULT_SYSTEM_PROMPT
)

# Initialize folders
ensure_folders_exist()

st.title("RAG Chatbot System")

# Sidebar Configuration
with st.sidebar:
    st.header("System Configuration")
    st.write(f"📄 Document: {PDF_FILE}")
    st.write(f"📚 Retrieval: SentenceTransformer")
    st.write(f"🔤 Embeddings Model: {MODEL_NAME}")
    st.write(f"🤖 Generator Model: {GENERATOR_MODEL}")
    
    st.header("Debug Settings")
    show_timings = st.checkbox("Show Processing Times", value=True)
    show_chunks = st.checkbox("Show Chunk Details", value=True)
    show_retrieved = st.checkbox("Show Retrieved Chunks", value=True)
    
    st.header("Retrieval Settings")
    k_chunks = st.slider(
        "Number of chunks to retrieve per query",
        min_value=1,
        max_value=5,
        value=2,
        help="Control how many relevant chunks to use for answering"
    )
    
    st.header("Prompt Configuration")
    system_prompt = st.text_area(
        "System Prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=100
    )
    user_prompt_template = "Question: {question}\nContext: {context}"

# Initialize Retriever
@st.cache_resource
def get_retriever():
    return initialize_retriever()

retriever, split_docs = get_retriever()

# Display chunks if enabled
if show_chunks:
    st.header("Document Analysis")
    st.metric("Total Chunks", len(split_docs))
    
    for i, chunk in enumerate(split_docs):
        with st.expander(f"📄 Chunk {i + 1}"):
            st.text(chunk.page_content)
            st.info(f"Characters: {len(chunk.page_content)}")

# Create the base chain
prompt, llm = create_chain(system_prompt, user_prompt_template)

# Debug retriever wrapper
def debug_retriever(retriever):
    def retrieve(query):
        st.divider()
        st.subheader("🔍 Retrieval Analysis")
        
        retrieval_start_time = time.time()
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start_time
        
        st.info(f"📊 Retrieved {len(docs)} most relevant chunks in {retrieval_time:.2f} seconds")
        st.write(f"💭 Query: '{query}'")
        
        for i, doc in enumerate(docs):
            with st.expander(f"📑 Chunk {i+1} of {len(docs)}", expanded=True):
                st.markdown("**Content:**")
                st.text(doc.page_content)
                st.markdown(f"""
                    **Analysis:**
                    - Characters: {len(doc.page_content)}
                    - Words: {len(doc.page_content.split())}
                    - Chunk used because it contains relevant information for the query
                """)
        
        st.divider()
        return docs
    
    return retrieve

# Update retriever with current k_chunks
retriever = update_retriever_k(retriever, k_chunks)
wrapped_retriever = debug_retriever(retriever)
chain = setup_chain(retriever, prompt, llm)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Display chat history
st.header("Chat History")
timestamp = time.time()

for i, entry in enumerate(st.session_state.chat_history):
    with st.expander(f"Q: {entry['question'][:50]}..."):
        st.text_area(
            "Question", 
            entry['question'], 
            height=100, 
            disabled=True,
            key=f"q_{i}_{timestamp}"
        )
        st.text_area(
            "Answer", 
            entry['response'], 
            height=150, 
            disabled=True,
            key=f"a_{i}_{timestamp}"
        )

# Question input and processing
st.header("Ask Your Question")
user_question = st.text_input("Enter your question:")

if user_question:
    start_time = time.time()
    try:
        with st.spinner("Generating answer..."):
            retrieved_docs = wrapped_retriever(user_question)
            
            response = chain.invoke(user_question)
            st.success("Answer:")
            st.write(response)
            
            st.session_state.chat_history.append({
                "question": user_question, 
                "response": response,
                "retrieved_chunks": [doc.page_content for doc in retrieved_docs]
            })
            save_chat_history(st.session_state.chat_history)
            
            if show_timings:
                st.info(f"⏱️ Total response time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.button("Clear History"):
    st.session_state.chat_history = []
    clear_chat_history()