"""
Main RAG orchestration module.
Handles LLM chains, document initialization, and coordinates embeddings and reranking.
"""

import os
import json
from typing import List, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document

from embeddings import EmbeddingRetriever, MODEL_NAME, check_and_rebuild_embeddings
from reranker import RerankingEmbeddingRetriever, USE_RERANKING
from utils import load_document_chunks, load_generator_prompt

# Load environment variables
load_dotenv()

# LLM Configurations
LLM_BASE_URL = "http://localhost:11434"
GENERATOR_MODEL = "llama3.2"
GEMINI_MODEL = "gemini-2.0-flash-exp"
DEFAULT_LLM_PROVIDER = "gemini"  # or "ollama"

# File configurations
PDF_FILE = "DORA.pdf"
CHUNKS_FILE = "dora_chunks_simple.json"
JSON_HISTORY_FILE = "chat_history.json"

# Default prompts
# DEFAULT_SYSTEM_PROMPT = """Answer the user's question using only the context provided. The answer should be precise without any extra detail that does not exist in the given context.
# If you don't understand the question or can't find relevant information in the retrieved context, reply with 'I don't know.'."""
DEFAULT_SYSTEM_PROMPT = load_generator_prompt()
DEFAULT_USER_PROMPT = "Question: {question}\nContext: {context}"

def initialize_retriever(
    use_reranking: bool = USE_RERANKING,
    k: int = 5,
    force_rebuild_embeddings: bool = False
) -> Tuple[RerankingEmbeddingRetriever, List[str]]:
    """
    Initialize the document retriever with embeddings and optional reranking.
    
    Args:
        use_reranking: Whether to use reranking
        k: Number of documents to retrieve
        force_rebuild_embeddings: Force rebuild of embeddings
    
    Returns:
        Tuple of (retriever, document_texts)
    """
    print("🚀 Initializing RAG retriever...")
    
    # Load document chunks
    try:
        texts = load_document_chunks(CHUNKS_FILE)
        print(f"📚 Loaded {len(texts)} document chunks")
    except Exception as e:
        print(f"❌ Failed to load document chunks: {e}")
        raise
    
    # Check if embeddings need rebuilding
    rebuilt = check_and_rebuild_embeddings(
        texts, 
        model_name=MODEL_NAME,
        force_rebuild=force_rebuild_embeddings
    )
    
    if rebuilt:
        print("🔧 Embeddings will be rebuilt during retriever initialization...")
    
    # Initialize retriever with reranking support
    try:
        retriever = RerankingEmbeddingRetriever(
            texts=texts,
            k=k,
            model_name=MODEL_NAME,
            use_reranking=use_reranking
        )
        
        # Print initialization summary
        embedding_info = retriever.get_embedding_info()
        reranking_info = retriever.get_reranking_info()
        
        print("Retriever initialized successfully!")
        print(f"- Embedding Model: {embedding_info['model_name']}")
        print(f"- Dimensions: {embedding_info['dimensions']}")
        print(f"- Documents: {embedding_info['document_count']}")
        print(f"- Reranking: {'Enabled' if reranking_info['use_reranking'] else 'Disabled'}")
        
        if reranking_info['use_reranking']:
            print(f"- Reranking Model: {reranking_info['reranking_model']}")
            print(f"- Initial retrieval: {reranking_info['initial_k']} docs")
            print(f"- Final output: {reranking_info['final_k']} docs")
        
        return retriever, texts
        
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        raise

def create_llm(
    provider: str = DEFAULT_LLM_PROVIDER,
    model_name: str = None,
    base_url: str = LLM_BASE_URL
):
    """Create and return an LLM instance based on provider."""
    try:
        if provider == "ollama":
            model = model_name or GENERATOR_MODEL
            llm = OllamaLLM(model=model, base_url=base_url)
            print(f"🤖 LLM initialized: Ollama {model}")
            return llm
            
        elif provider == "gemini":
            model = model_name or GEMINI_MODEL
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.0,
                convert_system_message_to_human=True
            )
            print(f"🤖 LLM initialized: Gemini {model}")
            return llm
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        raise

def create_chain(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT
):
    """Create the prompt chain for RAG."""
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt_template)
    ])
    
    return prompt

def setup_chain(retriever: RerankingEmbeddingRetriever, prompt, llm):
    """Set up the complete RAG chain."""
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def retrieve_and_format(query):
        docs = retriever.get_relevant_documents(query)
        return format_docs(docs)
    
    # Create the chain
    rag_chain = (
        RunnableParallel({
            "context": retrieve_and_format,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def update_retriever_k(retriever: RerankingEmbeddingRetriever, k: int):
    """Update the number of documents to retrieve."""
    retriever.set_k(k)
    return retriever

def update_retriever_reranking(
    retriever: RerankingEmbeddingRetriever, 
    use_reranking: bool, 
    initial_k: int = None
):
    """Update reranking settings."""
    retriever.set_reranking(use_reranking)
    if initial_k is not None:
        retriever.set_initial_k(initial_k)
    return retriever

def save_chat_history(query: str, response: str, docs: List[Document]):
    """Save chat interaction to history file."""
    try:
        # Load existing history
        if os.path.exists(JSON_HISTORY_FILE):
            with open(JSON_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new interaction
        interaction = {
            "query": query,
            "response": response,
            "retrieved_docs": [
                {
                    "content": doc.page_content,
                    "metadata": getattr(doc, 'metadata', {})
                }
                for doc in docs
            ],
            "timestamp": str(datetime.now())
        }
        
        history.append(interaction)
        
        # Save updated history
        with open(JSON_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"❌ Failed to save chat history: {e}")

def get_system_info() -> dict:
    """Get comprehensive system information."""
    return {
        "pdf_file": PDF_FILE,
        "chunks_file": CHUNKS_FILE,
        "embedding_model": MODEL_NAME,
        "generator_model": GENERATOR_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "use_reranking": USE_RERANKING,
        "history_file": JSON_HISTORY_FILE
    }

# Debug wrapper for retriever
class DebugRetrieverWrapper:
    """Wrapper to capture retrieval results for debugging."""
    
    def __init__(self, retriever: RerankingEmbeddingRetriever):
        self.retriever = retriever
        self.last_docs = []
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents and store for debugging."""
        self.last_docs = self.retriever.get_relevant_documents(query)
        return self.last_docs
    
    def get_last_retrieved_docs(self) -> List[Document]:
        """Get the last retrieved documents."""
        return self.last_docs

def debug_retriever(retriever: RerankingEmbeddingRetriever) -> DebugRetrieverWrapper:
    """Wrap retriever for debugging purposes."""
    return DebugRetrieverWrapper(retriever) 