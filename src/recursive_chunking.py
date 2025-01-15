import pdfplumber
import re
import json
from nltk.tokenize import sent_tokenize
import nltk


def clean_text(text: str) -> str:
    """Clean the text by removing headers and extra whitespace."""
    # Remove EU journal headers
    header_pattern = r'(?:27\.12\.2022 EN Official Journal of the European Union L 333/\d+|L 333/\d+ EN Official Journal of the European Union 27\.12\.2022)'
    text = re.sub(header_pattern, '', text, flags=re.MULTILINE)
    
    # Clean up whitespace
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
    # Common sentence endings
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
        # Find the end of the current chunk
        chunk_end = min(current_pos + chunk_size, text_length)
        
        # If we're not at the end of the text, find a proper sentence boundary
        if chunk_end < text_length:
            chunk_end = find_sentence_boundary(text, chunk_end, forward=True)
        
        # Extract the chunk
        chunk = text[current_pos:chunk_end].strip()
        
        # Only add if chunk meets minimum size
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        elif chunks:  # Append to previous chunk if too small
            chunks[-1] = chunks[-1] + " " + chunk
        
        # Move position for next chunk, accounting for overlap
        if chunk_end < text_length:
            # Find the start of the next chunk, considering overlap
            overlap_start = max(current_pos, chunk_end - overlap)
            current_pos = find_sentence_boundary(text, overlap_start, forward=False)
        else:
            current_pos = chunk_end
    
    # Validate chunks and ensure no broken sentences
    validated_chunks = []
    for chunk in chunks:
        # Remove any partial sentences at the start
        if not chunk[0].isupper() and validated_chunks:
            validated_chunks[-1] = validated_chunks[-1] + " " + chunk
            continue
            
        # Clean up the chunk
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
    
    # Extract and clean text
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += " " + clean_text(text)
    
    # Create chunks with specified parameters
    chunks = create_chunks(all_text, chunk_size=chunk_size, overlap=overlap, min_chunk_size=min_chunk_size)
    
    # Print chunk statistics
    print(f"\nChunking Statistics:")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} characters")
    print(f"Smallest chunk: {min(len(chunk) for chunk in chunks)} characters")
    print(f"Largest chunk: {max(len(chunk) for chunk in chunks)} characters")
    
    # Check overlaps
    if len(chunks) > 1:
        overlaps = []
        for i in range(len(chunks) - 1):
            words1 = set(chunks[i].split()[-10:])  # Last 10 words of current chunk
            words2 = set(chunks[i + 1].split()[:10])  # First 10 words of next chunk
            overlap_words = words1.intersection(words2)
            overlaps.append(len(overlap_words))
        
        print(f"\nOverlap Statistics:")
        print(f"Average overlapping words: {sum(overlaps) / len(overlaps):.1f}")
        print(f"Min overlapping words: {min(overlaps)}")
        print(f"Max overlapping words: {max(overlaps)}")
    
    # Optionally save to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\nChunks saved to: {output_file}")
    
    # Print sample chunks for verification
    print("\nSample chunks (first 2 chunks):")
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1} (Length: {len(chunk)}):")
        print("Start:", chunk[:100] + "...")
        print("End:", "..." + chunk[-100:])
    
    return chunks

if __name__ == "__main__":
    # Configuration
    PDF_PATH = "DORA.pdf"
    OUTPUT_FILE = "dora_chunks_simple.json"
    CHUNK_SIZE = 1000      # Target chunk size
    OVERLAP = 150          # Overlap between chunks
    MIN_CHUNK_SIZE = 200   # Minimum chunk size
    
    # Process the PDF with specified parameters
    chunks = process_pdf_for_rag(
        pdf_path=PDF_PATH,
        output_file=OUTPUT_FILE,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE
    )