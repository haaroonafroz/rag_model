import os
import json
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from utils import *
from transformers import AutoTokenizer
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
from utils import *

# Load environment variables
load_dotenv()

# Configurations
LLM_BASE_URL = "http://localhost:11434"  # Fixed: Ollama's default port
GENERATOR_MODEL = "llama3.2"
PDF_FILE = "DORA.pdf"
CHUNKS_FILE = "dora_chunks_simple.json"
VECTOR_EMBEDDINGS_FOLDER = "embeddings_index"
JSON_HISTORY_FILE = "chat_history.json"

# Embedding configurations
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
USE_GPU = False 
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

# Reranking configurations
RERANKING_MODEL = "BAAI/bge-reranker-base"  # Fast and effective cross-encoder
USE_RERANKING = True
INITIAL_RETRIEVAL_K = 20  # Retrieve more docs for reranking
RERANK_TOP_K = 5  # Final number after reranking

# Default prompts
DEFAULT_SYSTEM_PROMPT = """Answer the user's question using only the context provided. The answer should be precise without any extra detail that does not exist in the given context.
If you don't understand the question or can't find relevant information in the retrieved context, reply with 'I don't know.'."""
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

class RerankingEmbeddingRetriever(EmbeddingRetriever):
    def __init__(
        self,
        texts: List[str],
        k: int = 3,
        model_name: str = MODEL_NAME,
        reranking_model: str = RERANKING_MODEL,
        use_reranking: bool = USE_RERANKING,
        initial_k: int = INITIAL_RETRIEVAL_K,
        use_gpu: bool = USE_GPU
    ):
        # Initialize parent class
        super().__init__(texts, k, model_name, use_gpu)
        
        # Reranking configurations
        self.use_reranking = use_reranking
        self.initial_k = initial_k
        self.reranking_model = None
        
        # Initialize reranking model if enabled
        if self.use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self.reranking_model = CrossEncoder(reranking_model, device=self.device)
                print(f"✅ Reranking model loaded: {reranking_model}")
            except ImportError:
                print("❌ sentence-transformers not found. Install with: pip install sentence-transformers")
                self.use_reranking = False
            except Exception as e:
                print(f"❌ Failed to load reranking model: {e}")
                self.use_reranking = False
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve and rerank relevant documents for a query."""
        
        if not self.use_reranking:
            # Fall back to original retrieval
            return super().get_relevant_documents(query)
        
        # Step 1: Initial retrieval with higher k
        initial_k = max(self.initial_k, self.k)  # Ensure we retrieve at least k docs
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=self.device)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Search in FAISS index for initial candidates
        distances, indices = self.index.search(query_embedding, min(initial_k, len(self.texts)))
        
        # Get initial documents
        initial_docs = [
            {"text": self.texts[idx], "index": idx, "distance": distances[0][i]} 
            for i, idx in enumerate(indices[0]) if idx < len(self.texts)
        ]
        
        if len(initial_docs) <= self.k:
            # If we have few documents, return them directly
            return [Document(page_content=doc["text"]) for doc in initial_docs]
        
        # Step 2: Rerank using cross-encoder
        try:
            # Prepare query-document pairs for reranking
            query_doc_pairs = [[query, doc["text"]] for doc in initial_docs]
            
            # Get reranking scores
            rerank_scores = self.reranking_model.predict(query_doc_pairs)
            
            # Combine scores with documents
            for i, doc in enumerate(initial_docs):
                doc["rerank_score"] = float(rerank_scores[i])
            
            # Sort by reranking score (higher is better)
            reranked_docs = sorted(initial_docs, key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top-k reranked documents
            top_k_docs = reranked_docs[:self.k]
            
            return [Document(
                page_content=doc["text"],
                metadata={
                    "rerank_score": doc["rerank_score"],
                    "original_distance": doc["distance"],
                    "original_rank": next(i for i, d in enumerate(initial_docs) if d["index"] == doc["index"]) + 1
                }
            ) for doc in top_k_docs]
            
        except Exception as e:
            print(f"❌ Reranking failed, falling back to embedding similarity: {e}")
            # Fall back to original similarity-based ranking
            return [Document(page_content=doc["text"]) for doc in initial_docs[:self.k]]
    
    def set_reranking(self, use_reranking: bool):
        """Enable or disable reranking."""
        self.use_reranking = use_reranking and self.reranking_model is not None
    
    def set_initial_k(self, initial_k: int):
        """Set the number of documents to retrieve before reranking."""
        self.initial_k = initial_k

def initialize_retriever(use_reranking: bool = USE_RERANKING) -> Tuple[RerankingEmbeddingRetriever, List[Document]]:
    """Initialize or load the retriever and document chunks."""
    # Load or create chunks
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, 'r') as f:
            chunks = json.load(f)
    else:
        chunks = process_pdf_for_rag(PDF_FILE, CHUNKS_FILE)
    
    # Create document objects
    split_docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize retriever with reranking
    retriever = RerankingEmbeddingRetriever(
        texts=[doc.page_content for doc in split_docs],
        k=RERANK_TOP_K,
        use_reranking=use_reranking,
        initial_k=INITIAL_RETRIEVAL_K,
        use_gpu=USE_GPU
    )
    
    return retriever, split_docs

def create_chain(system_prompt: str = DEFAULT_SYSTEM_PROMPT, 
                user_prompt_template: str = DEFAULT_USER_PROMPT):
    """Create the LLM chain with specified prompts."""
    llm = OllamaLLM(model=GENERATOR_MODEL, base_url=LLM_BASE_URL)
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

def update_retriever_k(retriever: RerankingEmbeddingRetriever, k: int):
    """Update the number of documents to retrieve."""
    retriever.set_k(k)
    return retriever

def update_retriever_reranking(retriever: RerankingEmbeddingRetriever, use_reranking: bool, initial_k: int = None):
    """Update reranking settings."""
    retriever.set_reranking(use_reranking)
    if initial_k is not None:
        retriever.set_initial_k(initial_k)
    return retriever