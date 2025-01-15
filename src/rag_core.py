import os
import json
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from utils import *
from recursive_chunking import *
from transformers import AutoTokenizer
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configurations
LLM_BASE_URL = "http://localhost:11434"
GENERATOR_MODEL = "llama3.2"
PDF_FILE = "DORA.pdf"
CHUNKS_FILE = "dora_chunks_simple.json"
VECTOR_EMBEDDINGS_FOLDER = "embeddings_index"
JSON_HISTORY_FILE = "chat_history.json"

# Embedding configurations
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
USE_GPU = False  # Set to True if you have GPU
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using only the context provided. Make sure you Are not writing anything beyond the context
If you don't understand the question or can't find relevant information in the context, reply with 'I don't know sir '."""
DEFAULT_USER_PROMPT = "Question: {question}\nContext: {context}"

class EmbeddingRetriever:
    def __init__(
        self,
        texts: List[str],
        k: int = 3,
        model_name: str = MODEL_NAME,
        use_gpu: bool = USE_GPU
    ):
        self.k = k
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)
        self.texts = texts
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(VECTOR_EMBEDDINGS_FOLDER, exist_ok=True)
        
        # Initialize or load index
        self.index_path = os.path.join(VECTOR_EMBEDDINGS_FOLDER, "faiss.index")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.embeddings = np.load(os.path.join(VECTOR_EMBEDDINGS_FOLDER, "embeddings.npy"))
        else:
            self._create_index()
    
    def _create_index(self):
        """Create FAISS index from texts."""
        # Generate embeddings
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True, device=self.device)
        if isinstance(self.embeddings, torch.Tensor):
            self.embeddings = self.embeddings.cpu().numpy()
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        # Save index and embeddings
        faiss.write_index(self.index, self.index_path)
        np.save(os.path.join(VECTOR_EMBEDDINGS_FOLDER, "embeddings.npy"), self.embeddings)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=self.device)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, self.k)
        
        # Return relevant documents
        return [Document(page_content=self.texts[idx]) for idx in indices[0]]
    
    def set_k(self, k: int):
        """Update the number of documents to retrieve."""
        self.k = k

def ensure_folders_exist():
    """Ensure necessary folders and files exist."""
    os.makedirs(VECTOR_EMBEDDINGS_FOLDER, exist_ok=True)
    if not os.path.exists(JSON_HISTORY_FILE):
        with open(JSON_HISTORY_FILE, "w") as f:
            json.dump([], f)

def initialize_retriever() -> Tuple[EmbeddingRetriever, List[Document]]:
    """Initialize or load the retriever and document chunks."""
    # Load or create chunks
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, 'r') as f:
            chunks = json.load(f)
    else:
        chunks = process_pdf_for_rag(PDF_FILE, CHUNKS_FILE)
    
    # Create document objects
    split_docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize retriever
    retriever = EmbeddingRetriever(
        texts=[doc.page_content for doc in split_docs],
        k=3,
        use_gpu=USE_GPU
    )
    
    return retriever, split_docs

def create_chain(system_prompt: str = DEFAULT_SYSTEM_PROMPT, 
                user_prompt_template: str = DEFAULT_USER_PROMPT):
    """Create the LLM chain with specified prompts."""
    llm = Ollama(model=GENERATOR_MODEL, base_url=LLM_BASE_URL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", str(user_prompt_template))
    ])
    return prompt, llm

def setup_chain(retriever: EmbeddingRetriever, prompt, llm):
    """Set up the complete RAG chain."""
    setup = RunnableParallel(
        {"context": retriever.get_relevant_documents, "question": RunnablePassthrough()}
    )
    chain = setup | prompt | llm | StrOutputParser()
    return chain

def load_chat_history() -> List[dict]:
    """Load chat history from file."""
    if os.path.exists(JSON_HISTORY_FILE):
        with open(JSON_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(chat_history: List[dict]):
    """Save chat history to file."""
    with open(JSON_HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=4)

def clear_chat_history():
    """Clear the chat history."""
    with open(JSON_HISTORY_FILE, "w") as f:
        json.dump([], f)

def update_retriever_k(retriever: EmbeddingRetriever, k: int):
    """Update the number of documents to retrieve."""
    retriever.set_k(k)
    return retriever