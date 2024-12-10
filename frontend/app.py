import streamlit as st
from pipeline import RAGPipeline
from store_vector import VectorStore
from transformers import pipeline
from utils import generate_embeddings

st.title("PDF Q&A with RAG")

sample_chunk = ["This is a sample text."]
embedding_dim = len(generate_embeddings(sample_chunk)[0])

# Initialize components
qa_model = pipeline("question-answering")
vector_store = VectorStore(dim=embedding_dim)  # Assuming the embeddings have 384 dimensions
rag_pipeline = RAGPipeline(vector_store, qa_model)

# Upload PDF and ask questions
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Enter your question")

if uploaded_file and question:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load PDF and generate vectors
    rag_pipeline.load_pdf("temp.pdf")
    
if question:
    # Query the RAG pipeline with the question to fetch the answer with context
    answer = rag_pipeline.query(question)
    
    # Display the answer
    st.write("Answer:", answer)
