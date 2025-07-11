# utils.py
import os
import re
import json
import pdfplumber
import numpy as np
import faiss
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
# from rag_core import *


# Default prompts
DEFAULT_SYSTEM_PROMPT = """Answer the user's question using only the context provided. The answer should be precise without any extra detail that does not exist in the given context.
If you don't understand the question or can't find relevant information in the retrieved context, reply with 'I don't know.'."""

def load_generator_prompt(prompt_file: str = "src/generator_prompt.txt") -> str:
    """Load the generator prompt from a text file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ Prompt file {prompt_file} not found, using default prompt")
        return DEFAULT_SYSTEM_PROMPT
    except Exception as e:
        print(f"⚠️ Error loading prompt file: {e}, using default prompt")
        return DEFAULT_SYSTEM_PROMPT
# Utility Functions
def clean_text(text: str) -> str:
    """Clean the text by removing headers and extra whitespace."""
    header_pattern = r'(?:27\.12\.2022 EN Official Journal of the European Union L 333/\d+|L 333/\d+ EN Official Journal of the European Union 27\.12\.2022)'
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_sentence_boundary(text: str, position: int, forward: bool = True) -> int:
    """
    Find the nearest sentence boundary from a given position.
    
    Args:
        text: The text to search in
        position: Starting position
        forward: If True, search forward; if False, search backward
    
    Returns:
        Position of the nearest sentence boundary
    """
    endings = {'.', '!', '?', ':', ';'}
    if forward:
        for i in range(position, len(text)):
            if text[i] in endings and (i + 1 == len(text) or text[i + 1].isspace()):
                return i + 1
        return len(text)
    else:
        for i in range(position - 1, -1, -1):
            if i > 0 and text[i - 1] in endings and text[i].isspace():
                return i
        return 0

def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 50, min_chunk_size: int = 100) -> list:
    """
    Create chunks with proper word and sentence preservation.
    
    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for any chunk
    
    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    current_pos = 0
    text_length = len(text)

    while current_pos < text_length:
        chunk_end = min(current_pos + chunk_size, text_length)
        if chunk_end < text_length:
            chunk_end = find_sentence_boundary(text, chunk_end, forward=True)
        chunk = text[current_pos:chunk_end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        elif chunks:
            chunks[-1] += " " + chunk
        if chunk_end < text_length:
            overlap_start = max(current_pos, chunk_end - overlap)
            current_pos = find_sentence_boundary(text, overlap_start, forward=False)
        else:
            current_pos = chunk_end

    validated_chunks = []
    for chunk in chunks:
        if not chunk[0].isupper() and validated_chunks:
            validated_chunks[-1] += " " + chunk
            continue
        chunk = re.sub(r'\s+', ' ', chunk)
        validated_chunks.append(chunk)

    return validated_chunks

def process_pdf_for_rag(pdf_path: str, output_file: str = None, chunk_size: int = 1000, overlap: int = 50, min_chunk_size: int = 100):
    """
    Process PDF and create chunks suitable for RAG.
    
    Args:
        pdf_path: Path to PDF file
        output_file: Optional path to save chunks
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for any chunk
    """
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += " " + clean_text(text)

    chunks = create_chunks(all_text, chunk_size=chunk_size, overlap=overlap, min_chunk_size=min_chunk_size)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to: {output_file}")

    return chunks

def ensure_folders_exist():
    os.makedirs("embeddings_index", exist_ok=True)
    if not os.path.exists("chat_history.json"):
        with open("chat_history.json", "w") as f:
            json.dump([], f)

def load_chat_history() -> list:
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            return json.load(f)
    return []

def save_chat_history(question: str, response: str, retrieved_docs: list):
    """Save a single chat entry to the chat history file."""
    chat_history = load_chat_history()
    chat_history.append({
        "question": question, 
        "response": response,
        "retrieved_chunks": [doc.page_content for doc in retrieved_docs]
    })
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)

def clear_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump([], f)

def load_document_chunks(chunks_file: str) -> list:
    """Load document chunks from JSON file."""
    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_file}")
    
    return chunks
