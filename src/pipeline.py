from transformers import pipeline
from vector_store import VectorStore
from embeddings import generate_embeddings
from text_extraction import process_pdf_to_chunks

qa_pipeline = pipeline("question-answering")

class RAGPipeline:
    def __init__(self, vector_store, qa_model):
        self.vector_store = vector_store
        self.qa_model = qa_model

    def load_pdf(self, pdf_path):
        """Load PDF, convert to chunks, and store embeddings in the vector store."""
        chunks = process_pdf_to_chunks(pdf_path)
        embeddings = generate_embeddings(chunks)
        self.vector_store.add_vectors(embeddings, chunks)

    def query(self, question):
        """Query the vector store using the question and retrieve the answer."""
        question_embedding = generate_embeddings([question])[0]
        indices, _ = self.vector_store.query(question_embedding)
        context = " ".join([self.vector_store.get_chunk(i) for i in indices])
        return self.qa_model(context, question)['answer']
