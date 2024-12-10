import re
import nltk
from nltk.tokenize import sent_tokenize
import pdfplumber
# from utils import clean_text, chunk_text
from sentence_transformers import SentenceTransformer

def download_nltk_data():
    """Download necessary NLTK data."""
    nltk.download("punkt")


# -----------------Text Extraction-----------------------

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=50):
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    for sentence in sentences:
        if len(" ".join(chunk)) + len(sentence) <= max_length:
            chunk.append(sentence)
        else:
            chunks.append(" ".join(chunk))
            chunk = [sentence]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + " "
    return clean_text(text)

def process_pdf_to_chunks(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text)

# ------------------Embedding Generation------------------
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
model[0].default_task = 'retrieval.query'

def generate_embeddings(chunks):
    return model.encode(chunks, show_progress_bar=True)