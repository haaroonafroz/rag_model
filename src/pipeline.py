from transformers import pipeline
from store_vector import VectorStore
from utils import generate_embeddings
from utils import process_pdf_to_chunks

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
        """Retrieve the context and answer the question."""
        
        # Generate embedding for the question
        question_embedding = generate_embeddings([question])[0]
        
        # Retrieve relevant indices from the vector store (contextual information)
        indices, _ = self.vector_store.query(question_embedding)
        
        # Retrieve the corresponding context from your vector store (could be a document or chunk of text)
        context = " ".join([self.vector_store.get_chunk(i) for i in indices])
        
        # Prepare the inputs for the QA model
        inputs = {
            "question": question,
            "context": context
        }
        
        # Pass the inputs to the QA model
        result = self.qa_model(**inputs)
        
        # Extract the answer from the result and return it
        return result['answer']
