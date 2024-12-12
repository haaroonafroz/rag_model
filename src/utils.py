import re
import pdfplumber
import torch

from sentence_transformers import SentenceTransformer
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    strategy: str  # 'character', 'paragraph', or 'word'
    chunk_size: int
    overlap: Optional[int] = None

# -----------------Text Extraction-----------------------

def chunk_text(text: str, config: ChunkingConfig) -> List[str]:
    """Chunk text based on specified strategy."""
    if config.strategy == 'paragraph':
        # Split on double newlines and filter empty chunks
        paragraphs = [p.strip() for p in text.split('\n\n')]
        return [p for p in paragraphs if p]
    
    elif config.strategy == 'word':
        # Split text into words
        words = text.split()
        # Calculate step size based on overlap if provided
        step = config.chunk_size - (config.overlap or 0)
        chunks = []
        
        for i in range(0, len(words), step):
            chunk = words[i:i + config.chunk_size]
            if chunk:  # only add non-empty chunks
                chunks.append(' '.join(chunk))
        
        return chunks
        
    else:  # default to character-based
        # If no overlap specified, use chunk_size as step
        step = config.chunk_size - (config.overlap or 0)
        return [
            text[i:i + config.chunk_size].strip()
            for i in range(0, len(text), step)
            if text[i:i + config.chunk_size].strip()  # only keep non-empty chunks
        ]

def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages)
    return re.sub(r'\s+', ' ', text).strip()

# ------------------Embedding Generation------------------

class EmbeddingModel:
    _instance = None
    
    @classmethod
    def get_instance(cls) -> SentenceTransformer:
        """
        Singleton method to get or create the embedding model instance.
        Returns the same model instance throughout the program's lifetime.
        """
        if cls._instance is None:
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

def generate_embeddings(texts: List[str], task: str = 'retrieval.query', batch_size: int = 32) -> List[List[float]]:
    model = EmbeddingModel.get_instance()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True
    ).tolist()
        
    return embeddings
