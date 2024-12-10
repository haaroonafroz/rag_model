import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []  # Store chunks alongside the vectors

    def add_vectors(self, vectors, chunks):
        """Add vectors and their corresponding chunks to the vector store."""
        self.index.add(np.array(vectors))
        self.chunks.extend(chunks)

    def query(self, vector, top_k=5):
        """Query the vector store for the most similar vectors."""
        distances, indices = self.index.search(np.array([vector]), top_k)
        
        # Debugging: Print the type and content of indices
        print(f"indices (raw): {indices}")
        
        # Convert indices to a list of integers if needed (in case of a NumPy array)
        indices = indices.astype(int)  # Ensure indices are integers
        
        print(f"indices (as integers): {indices}")
        
        # Retrieve chunks corresponding to the indices, filtering out empty chunks
        context_chunks = [self.get_chunk(i) for i in indices.flatten()]
        context_chunks = [chunk for chunk in context_chunks if chunk]  # Filter out empty strings
        
        # Join only non-empty chunks
        context = " ".join(context_chunks)
        
        return context, distances


    def get_chunk(self, index):
        """Retrieve the chunk corresponding to the given index."""
        try:
            # Ensure index is an integer (only process valid indices)
            index = int(index)  # This will raise a ValueError if index is not convertible
        except (ValueError, TypeError) as e:
            print(f"Invalid index: {index} (Error: {e})")
            return ""  # Return an empty string instead of None for invalid indices
        
        # Make sure the index is within the range of available chunks
        if index < 0 or index >= len(self.chunks):
            print(f"Index {index} out of bounds.")
            return ""  # Return an empty string if index is out of bounds
    
        return self.chunks[index]