import re
import pdfplumber
import torch
import hashlib
import pickle

from sentence_transformers import SentenceTransformer
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    strategy: str  # 'character', 'paragraph', or 'word'
    chunk_size: int
    overlap: Optional[float] = None

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
        step = max(1, round(config.chunk_size - config.chunk_size * (config.overlap or 0)))
        chunks = []
        
        for i in range(0, len(words), step):
            chunk = words[i:i + config.chunk_size]
            if chunk:  # only add non-empty chunks
                chunks.append(' '.join(chunk))
        
        return chunks
        
    else:  # default to character-based
        # If no overlap specified, use chunk_size as step
        step = max(1, round(config.chunk_size - config.chunk_size * (config.overlap or 0)))
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

# ------------QA-Pipeline---------------
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForPreTraining, BertForQuestionAnswering

def truncate_context(context: str, max_tokens: int = 512) -> str:
    """
    Truncate the context to ensure it fits within the model's token limit.
    """
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    tokenized_context = tokenizer(context, truncation=True, max_length=max_tokens, return_tensors="pt")
    return tokenizer.decode(tokenized_context['input_ids'][0], skip_special_tokens=True)

# --------------------------------

def load_qa_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

QA_PIPELINE = load_qa_pipeline()

def generate_answer(query: str, context: str) -> str:

    """
    Generate an answer for a query based on the provided context using a generative language model.
    """    
    # Prepare input
    truncated_context = truncate_context(context, max_tokens=512)
    input_data = {"question": query, "context": truncated_context}
    # input_data = {"question": query, "context": context}
    
    # Generate an answer
    result = QA_PIPELINE(input_data)
    return result['answer']

# -----------------------------

# Load the summarization pipeline once and cache it for reuse
def load_summarization_pipeline():
    """
    Load the summarization pipeline with a generative model.
    """
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Cached instance of the summarizer
SUMMARIZER = load_summarization_pipeline()

# ------------------------------

def summarize_answer(chunks: List[str], query: str) -> str:
    """
    Summarize the retrieved chunks into a concise answer based on the query.
    
    Args:
        chunks (List[str]): Retrieved chunks of text.
        query (str): The user's question or query.
    
    Returns:
        str: A concise, query-focused summary.
    """
    # Combine the top-k chunks into a single context
    combined_context = " ".join(chunks)
    truncated_context = truncate_context(combined_context, max_tokens=1024)
    
    # Format input for summarization
    prompt = (
        f"Context: {truncated_context}\n\n"
        f"Question: {query}\n\n"
        "Provide a concise and relevant answer to the question based on the context."
    )
    
    # Generate the summary
    summary = SUMMARIZER(
        prompt,
        max_length=150,  # Limit the length of the summary
        min_length=50,   # Ensure the summary isn't too short
        do_sample=False  # Deterministic output
    )
    
    return summary[0]['summary_text']


# ----------Retrieve Embeddings-------------
# Function to save and load embeddings
def save_embeddings(file_name, embeddings, chunks):
    """Save embeddings and chunks to a file."""
    with open(f"embeddings_{file_name}.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "chunks": chunks}, f)

def load_embeddings(file_name):
    """Load embeddings and chunks from a file if available."""
    try:
        with open(f"embeddings_{file_name}.pkl", "rb") as f:
            data = pickle.load(f)
            return data["embeddings"], data["chunks"]
    except FileNotFoundError:
        return None, None

def get_file_hash(file) -> str:
    """Generate a hash for the file to uniquely identify it."""
    return hashlib.md5(file.getvalue()).hexdigest()