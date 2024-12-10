import streamlit as st
from pipeline import RAGPipeline

st.title("PDF Q&A with RAG")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Enter your question")

if uploaded_file and question:
    # Instantiate pipeline and process
    rag = RAGPipeline(...)  # Initialize with components
    rag.load_pdf(uploaded_file)
    answer = rag.query(question)
    st.write("Answer:", answer)
