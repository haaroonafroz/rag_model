import re
from nltk.tokenize import sent_tokenize
import pdfplumber
from utils import clean_text, chunk_text
from sentence_transformers import SentenceTransformer


# -----------------Text Extraction-----------------------
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
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    return model.encode(chunks, show_progress_bar=True)