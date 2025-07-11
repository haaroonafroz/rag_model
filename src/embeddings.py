"""
Embeddings module for RAG system.
Handles embedding generation, FAISS index management, and document retrieval.
"""

import os
import numpy as np
import torch
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Embedding configurations
MODEL_NAME = "BAAI/bge-large-en-v1.5"
USE_GPU = False
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
VECTOR_EMBEDDINGS_FOLDER = "embeddings_index"

class EmbeddingRetriever:
    """
    Base class for embedding-based document retrieval using FAISS.
    """
    
    def __init__(
        self,
        texts: List[str],
        k: int = 3,
        model_name: str = MODEL_NAME,
        use_gpu: bool = USE_GPU
    ):
        self.k = k
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.texts = texts
        
        # Initialize embedding model
        print(f"🔤 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name).to(self.device)
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(VECTOR_EMBEDDINGS_FOLDER, exist_ok=True)
        
        # Set up paths
        self.index_path = os.path.join(VECTOR_EMBEDDINGS_FOLDER, "faiss.index")
        self.embeddings_path = os.path.join(VECTOR_EMBEDDINGS_FOLDER, "embeddings.npy")
        self.model_info_path = os.path.join(VECTOR_EMBEDDINGS_FOLDER, "model_info.txt")
        
        # Load or create embeddings
        self._load_or_create_embeddings()
    
    def _embeddings_exist_and_valid(self) -> bool:
        """Check if embeddings exist and are from the same model."""
        if not (os.path.exists(self.index_path) and 
                os.path.exists(self.embeddings_path) and 
                os.path.exists(self.model_info_path)):
            return False
        
        # Check if model matches
        try:
            with open(self.model_info_path, 'r') as f:
                stored_model = f.read().strip()
            return stored_model == self.model_name
        except:
            return False
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones."""
        if self._embeddings_exist_and_valid():
            print("📚 Loading existing embeddings...")
            self._load_embeddings()
        else:
            print("🔧 Creating new embeddings (this may take a minute)...")
            self._create_embeddings()
    
    def _load_embeddings(self):
        """Load existing FAISS index and embeddings."""
        try:
            self.index = faiss.read_index(self.index_path)
            self.embeddings = np.load(self.embeddings_path)
            print(f"✅ Loaded embeddings: {self.embeddings.shape[0]} documents, {self.embeddings.shape[1]} dimensions")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            print("🔧 Creating new embeddings...")
            self._create_embeddings()
    
    def _create_embeddings(self):
        """Create new FAISS index from texts."""
        try:
            # Generate embeddings
            print(f"📊 Encoding {len(self.texts)} documents...")
            self.embeddings = self.model.encode(
                self.texts, 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=True
            )
            
            if isinstance(self.embeddings, torch.Tensor):
                self.embeddings = self.embeddings.cpu().numpy()
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            print(f"🔍 Creating FAISS index with {dimension} dimensions...")
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            
            # Save everything
            self._save_embeddings()
            print(f"✅ Created embeddings: {self.embeddings.shape[0]} documents, {dimension} dimensions")
            
        except Exception as e:
            print(f"❌ Failed to create embeddings: {e}")
            raise
    
    def _save_embeddings(self):
        """Save FAISS index, embeddings, and model info."""
        faiss.write_index(self.index, self.index_path)
        np.save(self.embeddings_path, self.embeddings)
        
        # Save model info for validation
        with open(self.model_info_path, 'w') as f:
            f.write(self.model_name)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=self.device)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, self.k)
        
        # Return relevant documents
        return [
            Document(
                page_content=self.texts[idx],
                metadata={"distance": float(distances[0][i]), "index": int(idx)}
            ) 
            for i, idx in enumerate(indices[0]) if idx < len(self.texts)
        ]
    
    def set_k(self, k: int):
        """Update the number of documents to retrieve."""
        self.k = k
    
    def get_embedding_info(self) -> dict:
        """Get information about the current embeddings."""
        return {
            "model_name": self.model_name,
            "dimensions": self.embeddings.shape[1] if hasattr(self, 'embeddings') else None,
            "document_count": len(self.texts),
            "device": self.device,
            "index_path": self.index_path
        }

def check_and_rebuild_embeddings(
    texts: List[str], 
    model_name: str = MODEL_NAME,
    force_rebuild: bool = False
) -> bool:
    """
    Check if embeddings need to be rebuilt.
    Returns True if embeddings were rebuilt, False if existing ones were used.
    """
    embeddings_folder = VECTOR_EMBEDDINGS_FOLDER
    model_info_path = os.path.join(embeddings_folder, "model_info.txt")
    
    # Check if force rebuild
    if force_rebuild:
        print("🔄 Force rebuild requested...")
        _clean_embeddings_folder()
        return True
    
    # Check if embeddings exist
    if not os.path.exists(embeddings_folder):
        print("📁 Embeddings folder doesn't exist...")
        return True
    
    # Check if model info exists and matches
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r') as f:
                stored_model = f.read().strip()
            if stored_model != model_name:
                print(f"🔄 Model changed: {stored_model} → {model_name}")
                _clean_embeddings_folder()
                return True
        except:
            print("❌ Model info file corrupted...")
            _clean_embeddings_folder()
            return True
    else:
        print("📄 Model info missing...")
        return True
    
    print(f"✅ Using existing embeddings for {model_name}")
    return False

def _clean_embeddings_folder():
    """Clean the embeddings folder."""
    import shutil
    if os.path.exists(VECTOR_EMBEDDINGS_FOLDER):
        shutil.rmtree(VECTOR_EMBEDDINGS_FOLDER)
    os.makedirs(VECTOR_EMBEDDINGS_FOLDER, exist_ok=True) 