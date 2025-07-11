import json
import logging
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from embeddings import EmbeddingRetriever
import random
import os

cwd = os.getcwd()

# -----------------------------------------------------------------------------------------------------
# Custom prompts for different question types
factual_prompt = """
Context: {context}
Generate a single, fact-based question that focuses on specific facts or numbers mentioned in the context. The question should be concise and answerable using only the information provided.
Question {question_num}:
"""

conceptual_prompt = """
Context: {context}
Generate a single, concept-based question that tests understanding of concepts or relationships described in the context. The question should be concise and focused on 'why' or 'how' rather than simple facts.
Question {question_num}:
"""

# Default prompt if none provided
default_prompt = """
Context: {context}
Generate a single, concise question based on the context provided. Ensure the question is clear, specific, and answerable using only the information in the context.
Question {question_num}:
"""
# ------------------------------------------------------------------------------------------------------

#================ Configuration Constants============================
CHUNKS_FILE = "dora_chunks_simple.json"
# QUESTION_CONTEXT_PAIRS_FILE = "question_context_pairs.json"
# LOG_FILE = "evaluation_log.txt"
GENERATOR_MODEL = "llama3.2"
LLM_BASE_URL = "http://localhost:11434"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
USE_GPU = True
prompt_type = 'default' # 'conceptual' or 'factual'

k = 5 # select number of questions 
# ------------------------------------------------------------------------------------------------------


def setup_question_generator(model_type="ollama", model_name="llama3.2"):
    """Set up different types of language models for question generation."""
    if model_type == "ollama":
        return Ollama(model=model_name, base_url=LLM_BASE_URL)
    # Add other model types as needed


def get_log_filename(k, prompt_type="default"):
    """
    Generate a unique log file name based on k, prompt_type, and log count.
    """
    log_dir = cwd + '/Logs'
    # Create the Logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_count = 1
    while os.path.exists(os.path.join(log_dir, f"evaluation_log_k{k}_{prompt_type}_{log_count}.txt")):
        log_count += 1
    return os.path.join(log_dir, f"evaluation_log_k{k}_{prompt_type}_{log_count}.txt")

# -------------------------------------------------------------------------------
LOG_FILE = get_log_filename(k, prompt_type=prompt_type)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# -------------------------------------------------------------------------------

def load_chunks_from_file(chunks_file):
    """Load chunks from a JSON file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

import random

def generate_question_context_pairs_from_chunks(chunks, llm, num_pairs=50, questions_per_chunk=1, custom_prompt=None, prompt_type='default'):
    """
    Generate question-context pairs with customizable parameters.
    
    Args:
        chunks: List of text chunks to generate questions from.
        llm: Language model instance.
        num_pairs: Number of chunks to use.
        questions_per_chunk: Number of questions to generate per chunk.
        custom_prompt: Optional custom prompt for question generation.
        prompt_type: Type of prompt used (e.g., 'factual', 'conceptual').
    
    Returns:
        List of dictionaries containing context-question pairs.
    """
    # Dynamically name the output file based on the prompt type
    output_file = f"question_context_pairs_{prompt_type}.json"
    

    # Check if the JSON file with question-context pairs already exists
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
            logging.info(f"Loaded existing question-context pairs from {output_file}")
            return pairs
    except FileNotFoundError:
        logging.info(f"{output_file} not found. Generating new pairs.")
    
    # Ensure the JSON file exists
    open(output_file, 'a').close()
    
    # Randomly select chunks for processing
    selected_chunks = random.sample(chunks, min(num_pairs, len(chunks)))
    
    pairs = []
    
    generation_prompt = custom_prompt if custom_prompt else default_prompt
    
    for i, chunk in enumerate(selected_chunks):
        for j in range(questions_per_chunk):
            prompt = generation_prompt.format(
                context=chunk,
                question_num=j + 1
            )
            question = llm.invoke(prompt).strip()
            # if "\n" in question:
            #     question = question.split("\n")[0].strip()
            pairs.append({
                "context": chunk,
                "question": question.strip(),
                "chunk_index": i,
            })
    
    # Save pairs to the dynamically named JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=4)
    logging.info(f"Generated {len(pairs)} question-context pairs and saved to {output_file}")
    return pairs


def initialize_retriever_with_chunks(chunks, model_name, k, use_gpu=True):
    """Initialize the retriever with chunks from the JSON file."""
    retriever = EmbeddingRetriever(
        texts=chunks,
        k=k,
        model_name=model_name,
        use_gpu=use_gpu
    )
    return retriever

def evaluate_retrieval(retriever, question_context_pairs, k):
    """
    Evaluate the retrieval model with hit rate and MRR metrics.
    """
    hits = 0
    reciprocal_ranks = []

    for pair in question_context_pairs:
        question = pair["question"]
        ground_truth = pair["context"]

        # Retrieve top-k chunks
        retrieved_docs = retriever.get_relevant_documents(question)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        # Log retrieved chunks
        logging.info(f"Query: {question}: Generated from Chunk: {pair['context'][:100]}...")
        logging.info("Retrieved Chunks:")
        for idx, text in enumerate(retrieved_texts):
            logging.info(f"  Chunk {idx + 1}: {text[:100]}...")

        # Check if ground truth is in retrieved texts
        if ground_truth in retrieved_texts:
            hits += 1
            rank = retrieved_texts.index(ground_truth) + 1
            reciprocal_ranks.append(1 / rank)
            logging.info("Result: HIT")
        else:
            reciprocal_ranks.append(0)
            logging.info("Result: MISS")

    # Compute metrics
    hit_rate = hits / len(question_context_pairs)
    mrr = sum(reciprocal_ranks) / len(question_context_pairs)

    logging.info(f"Hit Rate: {hit_rate:.2f}")
    logging.info(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")

    return hit_rate, mrr

def evaluate_retrieval_with_chunks(chunks_file, retriever, llm, num_pairs=50, k=k, prompt_type= prompt_type):
    """
    Full evaluation pipeline using chunks loaded from a JSON file.
    """
    logging.info(f"Starting evaluation with k={k} and prompt_type={prompt_type}")
    
    # Load chunks
    chunks = load_chunks_from_file(chunks_file)

    # Choose the appropriate prompt and generate question-context pairs
    custom_prompt = factual_prompt if prompt_type == "factual" else conceptual_prompt
    question_context_pairs = generate_question_context_pairs_from_chunks(
        chunks=chunks,
        llm=llm,
        num_pairs=num_pairs,
        questions_per_chunk=1,  # Adjust as needed
        custom_prompt=custom_prompt,
        prompt_type=prompt_type
    )

    # Evaluate retrieval
    hit_rate, mrr = evaluate_retrieval(retriever, question_context_pairs, k)

    return hit_rate, mrr

if __name__ == "__main__":
    # Initialize LLM with more control
    llm = setup_question_generator(
        model_type="ollama",
        model_name=GENERATOR_MODEL
    )

    # Load chunks and initialize retriever
    chunks = load_chunks_from_file(CHUNKS_FILE)
    retriever = initialize_retriever_with_chunks(
        chunks, model_name=MODEL_NAME, k=k, use_gpu=USE_GPU
    )

    # Choose prompt type ('factual' or 'conceptual') and evaluate
    # prompt_type = "conceptual"  # Change to "conceptual" for conceptual questions
    hit_rate, mrr = evaluate_retrieval_with_chunks(
        CHUNKS_FILE,
        retriever,
        llm,
        num_pairs= 100,  # Adjust as needed
        k=k,
        prompt_type=prompt_type
    )

    # Print metrics
    print(f"Hit Rate: {hit_rate:.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")
