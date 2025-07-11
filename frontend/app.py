import streamlit as st
import time
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import *
from main import (
    initialize_retriever, create_llm, create_chain, setup_chain,
    update_retriever_k, update_retriever_reranking, save_chat_history,
    get_system_info, debug_retriever, DEFAULT_SYSTEM_PROMPT,
    GENERATOR_MODEL, LLM_BASE_URL
)
from embeddings import MODEL_NAME
from reranker import RERANKING_MODEL

# Initialize folders
ensure_folders_exist()

st.title("RAG Chatbot System")

# Get system info
system_info = get_system_info()

# Sidebar Configuration
with st.sidebar:
    st.header("System Configuration")
    st.write(f"-->Document: {system_info['pdf_file']}")
    st.write(f"-->Retriever Model: SentenceTransformer + Cross-Encoder")
    st.write(f"-->Embeddings Model: {system_info['embedding_model']}")
    st.write(f"-->Reranking Model: {RERANKING_MODEL}")
    st.write(f"-->Generator Model: {system_info['generator_model']}")
    
    st.header("Debug Settings")
    show_timings = st.checkbox("Show Processing Times", value=True)
    show_chunks = st.checkbox("Show Chunk Details", value=True)
    show_retrieved = st.checkbox("Show Retrieved Chunks", value=True)
    
    st.header("🤖 Model Settings")
    llm_provider = st.selectbox(
        "Choose LLM Provider",
        options=["ollama", "gemini"],
        index=0,
        help="Select between local Ollama or cloud-based Gemini"
    )
    
    if llm_provider == "ollama":
        ollama_model = st.selectbox(
            "Ollama Model",
            options=["llama3.2", "llama2", "codellama", "mistral"],
            index=0,
            help="Local Ollama model to use"
        )
        st.info("💻 Using local Ollama - requires Ollama to be running")
        
    elif llm_provider == "gemini":
        gemini_model = st.selectbox(
            "Gemini Model", 
            options=["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
            index=0,
            help="Google Gemini model to use"
        )
        st.info("☁️ Using Google Gemini - requires API key in .env file")
    
    st.divider()
    
    st.header("Retrieval Settings")
    k_chunks = st.slider(
        "Number of chunks to retrieve per query",
        min_value=1,
        max_value=50,
        value=10,
        help="Control how many relevant chunks to use for answering"
    )
    
    st.header("🎯 Reranking Settings")
    use_reranking = st.checkbox(
        "Enable Reranking", 
        value=True,
        help="Use cross-encoder reranking for better retrieval quality"
    )
    
    if use_reranking:
        initial_k = st.slider(
            "Initial retrieval count (before reranking)",
            min_value=k_chunks,
            max_value=50,
            value=20,
            help="How many documents to retrieve before reranking"
        )
        st.info(f"🔄 Pipeline: Retrieve {initial_k} → Rerank → Top {k_chunks}")
    else:
        initial_k = k_chunks
        st.info(f"📊 Pipeline: Retrieve {k_chunks} (no reranking)")
    
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
    return initialize_retriever(use_reranking=True)

@st.cache_resource  
def get_llm(provider, model_name=None):
    return create_llm(provider=provider, model_name=model_name)

retriever, split_docs = get_retriever()

# Initialize LLM based on selection
if llm_provider == "ollama":
    llm = get_llm("ollama", ollama_model)
elif llm_provider == "gemini":
    llm = get_llm("gemini", gemini_model)

# Create the base chain
prompt = create_chain(system_prompt, user_prompt_template)

# Debug retriever wrapper
def debug_retriever(retriever):
    def retrieve(query):
        st.divider()
        st.subheader("🔍 Retrieval Analysis")
        
        retrieval_start_time = time.time()
        
        # Check if reranking is enabled
        has_reranking = hasattr(retriever, 'use_reranking') and retriever.use_reranking
        
        if has_reranking:
            st.info(f"📊 Stage 1: Initial retrieval from embeddings index...")
            
            # Get initial retrieval count
            initial_k = getattr(retriever, 'initial_k', 20)
            st.write(f"💭 Query: '{query}'")
            st.write(f"🎯 Retrieving {initial_k} candidates for reranking...")
            
            # Add a placeholder for reranking status
            reranking_status = st.empty()
            reranking_status.warning("🔄 Reranking in progress...")
            
        else:
            st.write(f"💭 Query: '{query}'")
            st.write("📊 Using direct embedding similarity (no reranking)")
        
        # Execute the actual retrieval
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start_time
        
        # Clear the reranking status if it was shown
        if has_reranking:
            reranking_status.empty()
        
        # Check if any docs have reranking metadata
        reranked_docs = [doc for doc in docs if hasattr(doc, 'metadata') and doc.metadata.get('rerank_score') is not None]
        
        if has_reranking and reranked_docs:
            # Show reranking process
            st.success(f"🎯 Stage 2: Reranking completed! Selected top {len(docs)} from {initial_k} candidates")
            st.info(f"⏱️ Total retrieval + reranking time: {retrieval_time:.2f} seconds")
            
            # Show reranked results
            st.subheader("🏆 Final Reranked Results")
            
            for i, doc in enumerate(docs):
                metadata = getattr(doc, 'metadata', {})
                rerank_score = metadata.get('rerank_score', 0)
                original_rank = metadata.get('initial_rank', i + 1)
                rank_change = original_rank - (i + 1)
                
                # Create title with rerank info
                title = f"🥇 Rank #{i+1} (🎯 Score: {rerank_score:.3f})"
                if rank_change > 0:
                    title += f" ↗️ +{rank_change}"
                elif rank_change < 0:
                    title += f" ↘️ {rank_change}"
                else:
                    title += " ➡️ 0"
                
                title += f" | Originally #{original_rank}"
                
                with st.expander(title, expanded=(i < 3)):  # Expand top 3
                    st.markdown("**Content:**")
                    st.text(doc.page_content)
                    
                    # Create detailed analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Document Stats:**")
                        st.write(f"• Characters: {len(doc.page_content)}")
                        st.write(f"• Words: {len(doc.page_content.split())}")
                        st.write(f"• Original embedding rank: #{original_rank}")
                    
                    with col2:
                        st.markdown("**🎯 Reranking Results:**")
                        st.write(f"• Rerank score: {rerank_score:.4f}")
                        st.write(f"• Final rank: #{i + 1}")
                        if rank_change > 0:
                            st.write(f"• ↗️ Promoted by {rank_change} positions")
                        elif rank_change < 0:
                            st.write(f"• ↘️ Demoted by {abs(rank_change)} positions")
                        else:
                            st.write(f"• ➡️ Rank unchanged")
                    
                    # Quality indicator
                    if rerank_score > 0.5:
                        st.success("🟢 High relevance - Strong semantic match")
                    elif rerank_score > 0.0:
                        st.warning("🟡 Medium relevance - Moderate semantic match") 
                    else:
                        st.error("🔴 Low relevance - Weak semantic match")
        
        elif has_reranking and not reranked_docs:
            # Reranking was enabled but failed or no metadata
            st.warning("⚠️ Reranking was enabled but failed - showing embedding similarity results")
            st.info(f"📊 Retrieved {len(docs)} chunks using embedding similarity in {retrieval_time:.2f} seconds")
            
            for i, doc in enumerate(docs):
                title = f"📑 Chunk {i+1} of {len(docs)} (Fallback: Embedding Only)"
                
                with st.expander(title, expanded=True):
                    st.markdown("**Content:**")
                    st.text(doc.page_content)
                    
                    st.markdown(f"""
                        **Analysis:**
                        - Characters: {len(doc.page_content)}
                        - Words: {len(doc.page_content.split())}
                        - Rank: #{i + 1} (embedding similarity fallback)
                        - Note: Reranking failed, using original embedding ranking
                    """)
        
        else:
            # Show regular embedding-only results
            st.info(f"📊 Retrieved {len(docs)} chunks using embedding similarity in {retrieval_time:.2f} seconds")
            
            if not has_reranking:
                st.info("ℹ️ Reranking is disabled - showing direct embedding similarity results")
            
            for i, doc in enumerate(docs):
                metadata = getattr(doc, 'metadata', {})
                distance = metadata.get('distance', 0)
                
                title = f"📑 Chunk {i+1} of {len(docs)}"
                if distance:
                    title += f" (Distance: {distance:.3f})"
                
                with st.expander(title, expanded=True):
                    st.markdown("**Content:**")
                    st.text(doc.page_content)
                    
                    analysis = f"""
                        **Analysis:**
                        - Characters: {len(doc.page_content)}
                        - Words: {len(doc.page_content.split())}
                        - Embedding similarity rank: #{i + 1}
                        - Selection: Direct embedding similarity (no reranking)
                    """
                    
                    if distance:
                        analysis += f"""
                        - Distance score: {distance:.4f}
                        """
                    
                    st.markdown(analysis)
        
        st.divider()
        return docs
    
    return retrieve

# Update retriever with current settings
retriever = update_retriever_k(retriever, k_chunks)
retriever = update_retriever_reranking(retriever, use_reranking, initial_k)
wrapped_retriever = debug_retriever(retriever)
chain = setup_chain(retriever, prompt, llm)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# Display chat history
st.header("Chat History")
timestamp = time.time()

for i, entry in enumerate(st.session_state.chat_history):
    # Handle both 'question' and 'query' keys for backward compatibility
    question_text = entry.get('question', entry.get('query', 'No question available'))
    with st.expander(f"Q: {question_text[:50]}..."):
        st.text_area(
            "Question", 
            question_text, 
            height=100, 
            disabled=True,
            key=f"q_{i}_{timestamp}"
        )
        st.text_area(
            "Answer", 
            entry.get('response', 'No response available'), 
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
            
            # Save to file and update session state
            save_chat_history(user_question, response, retrieved_docs)
            st.session_state.chat_history.append({
                "question": user_question, 
                "response": response,
                "retrieved_chunks": [doc.page_content for doc in retrieved_docs]
            })
            
            if show_timings:
                st.info(f"⏱️ Total response time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.button("Clear History"):
    st.session_state.chat_history = []
    clear_chat_history()

    # Display chunks if enabled
if show_chunks:
    st.header("Document Analysis")
    st.metric("Total Chunks", len(split_docs))
    
    for i, chunk in enumerate(split_docs):
        with st.expander(f"📄 Chunk {i + 1}"):
            st.text(chunk)  # chunk is already a string, not a Document object
            st.info(f"Characters: {len(chunk)}")