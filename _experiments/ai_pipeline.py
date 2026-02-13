import torch
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from cache_manager import embedding_cache
from typing import Tuple, Optional

def load_embedding_model():
    """Loads the Sentence Transformer model onto the GPU if available."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return model

import numpy as np

def generate_embeddings(chunks: list[str], model: SentenceTransformer, use_cache: bool = True) -> np.ndarray:
    """
    Generates embeddings for a list of text chunks with caching support.
    
    Args:
        chunks: List of text chunks
        model: SentenceTransformer model
        use_cache: Whether to use cached embeddings if available
        
    Returns:
        Numpy array of embeddings
    """
    # Try to use cache if enabled
    if use_cache:
        # Create a cache key from all chunks combined
        combined_text = "\n".join(chunks)
        cached_embeddings = embedding_cache.get_embeddings(combined_text)
        if cached_embeddings is not None:
            return cached_embeddings
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, 
                              convert_to_tensor=True, 
                              show_progress_bar=True)
    # FAISS requires the embeddings to be on the CPU as a numpy array
    embeddings_np = embeddings.cpu().numpy()
    
    # Cache the embeddings if caching is enabled
    if use_cache:
        combined_text = "\n".join(chunks)
        embedding_cache.save_embeddings(combined_text, embeddings_np)
    
    return embeddings_np

import faiss

def create_faiss_index(embeddings: np.ndarray):
    """Creates a FAISS index from a set of embeddings."""
    # The dimension of our vectors is 384 for the 'all-MiniLM-L6-v2' model
    dimension = embeddings.shape[1]
    # We use IndexFlatL2 for exact, fast search
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def get_or_create_faiss_index(
    text_chunks: list[str], 
    embedding_model: SentenceTransformer,
    use_cache: bool = True
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Get FAISS index from cache or create new one with embeddings.
    
    Args:
        text_chunks: List of text chunks
        embedding_model: SentenceTransformer model
        use_cache: Whether to use cached index if available
        
    Returns:
        Tuple of (FAISS index, embeddings array)
    """
    # Try to use cache if enabled
    if use_cache:
        combined_text = "\n".join(text_chunks)
        cached_result = embedding_cache.get_faiss_index(combined_text)
        if cached_result is not None:
            cached_index, cached_chunks = cached_result
            # Verify chunks match
            if cached_chunks == text_chunks:
                # Regenerate embeddings from index (FAISS doesn't store them)
                embeddings = np.array([cached_index.reconstruct(i) for i in range(cached_index.ntotal)])
                return cached_index, embeddings
    
    # Generate new embeddings and index
    embeddings = generate_embeddings(text_chunks, embedding_model, use_cache=use_cache)
    index = create_faiss_index(embeddings)
    
    # Cache the index if caching is enabled
    if use_cache:
        combined_text = "\n".join(text_chunks)
        embedding_cache.save_faiss_index(combined_text, index, text_chunks)
    
    return index, embeddings

# --- This is our test block for Milestone 2 ---

from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_custom_model_and_tokenizer(model_path: str, device: str):
    """Loads the custom fine-tuned model and tokenizer."""
    print(f"Loading custom model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer


def extractive_summary(text: str, summary_percentage: float) -> str:
    """Creates an extractive summary of a text."""
    sentences = nltk.sent_tokenize(text)
    if not sentences: return ""
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Handle case where text is only stopwords
        return ""
        
    sentence_scores = tfidf_matrix.sum(axis=1)
    ranked_indices = sentence_scores.argsort(axis=0)[::-1]
    num_sentences = int(len(sentences) * summary_percentage)
    if num_sentences == 0 and len(sentences) > 0:
        num_sentences = 1 # Ensure at least one sentence for very short texts
        
    selected_indices = sorted(ranked_indices[:num_sentences].A1)
    summary = " ".join([sentences[i] for i in selected_indices])
    return summary



def answer_question(question: str, text_chunks: list[str], faiss_index, embedding_model, llm_model, llm_tokenizer, device: str) -> str:
    """
    Answers a question using the RAG pipeline.
    
    Args:
        question (str): The user's question.
        text_chunks (list[str]): The original list of text chunks.
        faiss_index: The FAISS index of the text chunks.
        embedding_model: The loaded sentence-transformer model.
        llm_model: The loaded generative LLM (your custom model).
        llm_tokenizer: The tokenizer for the LLM.
        device (str): The device to run on ('cuda' or 'cpu').

    Returns:
        str: The generated answer.
    """
    # --- Step A: Retrieve ---
    # 1. Embed the user's question
    question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()

    # 2. Search the FAISS index for the top 5 most relevant chunks
    distances, indices = faiss_index.search(question_embedding, k=5)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)
    
    # --- Step B: Generate ---
    # 3. Create a prompt with the context and question
    prompt = f"""
    Context:
    {context}
    
    Based on the context provided, please answer the following question.
    
    Question: {question}
    Answer:
    """

    # 4. Tokenize the prompt and generate the answer
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

def generate_quiz(context: str, llm_model, llm_tokenizer, device: str, num_questions: int = 3) -> str:
    """
    Generates a multiple-choice quiz from a given context.
    The number of questions is now a parameter.

    Args:
        context (str): The text to generate the quiz from.
        llm_model: The loaded generative LLM (flan-t5-base).
        llm_tokenizer: The tokenizer for the LLM.
        device (str): The device to run on ('cuda' or 'cpu').
        num_questions (int): The number of questions to generate.

    Returns:
        str: The generated quiz.
    """
    # Create a detailed prompt for the model that uses the new parameter
    prompt = f"""
    Based on the following context, generate a multiple-choice quiz with {num_questions} questions.
    Each question should have 4 options (A, B, C, D).
    Clearly mark the correct answer with an asterisk (*) at the end of the correct option.

    Context:
    {context}

    Quiz:
    """

    # Tokenize the prompt and generate the quiz
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=512, num_beams=4, early_stopping=True)
    quiz = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return quiz

def generate_explanation(context: str, question: str, correct_answer: str, llm_model, llm_tokenizer, device: str) -> str:
    """
    Generates an explanation for why a quiz answer is correct.

    Args:
        context (str): The text the quiz was based on.
        question (str): The quiz question.
        correct_answer (str): The correct answer to the question.
        llm_model: The loaded generative LLM (flan-t5-base).
        llm_tokenizer: The tokenizer for the LLM.
        device (str): The device to run on ('cuda' or 'cpu').

    Returns:
        str: The generated explanation.
    """
    prompt = f"""
    Based on the context provided, please provide a brief, one-sentence explanation for why the answer is correct.

    Context:
    "{context}"

    Question:
    "{question}"

    Correct Answer:
    "{correct_answer}"

    Explanation:
    """

    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=100, num_beams=2, early_stopping=True)
    explanation = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return explanation


# --- This is our test block for Milestones 3 & 4 ---
if __name__ == '__main__':
    # Import functions
    from processing import extract_text, chunk_text

    # --- Setup: Load models and process a document ---
    print("--- Setup: Loading models and processing document ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the instruction-tuned model for Q&A and Quizzes
    model_name = "google/flan-t5-base"
    llm_model, llm_tokenizer = load_custom_model_and_tokenizer(model_name, device)
    
    # Load the embedding model for retrieval
    embedding_model = load_embedding_model()
    
    # Process the document
    text_chunks = chunk_text(extract_text('sample.txt'))
    embeddings = generate_embeddings(text_chunks, embedding_model)
    faiss_index = create_faiss_index(embeddings)
    print("\n--- Setup Complete ---")

    # --- Test Milestone 3: RAG Q&A ---
    user_question = "Explain Milestone 5?"
    print(f"\n--- Testing RAG Q&A ---")
    print(f"User Question: {user_question}")
    answer = answer_question(user_question, text_chunks, faiss_index, embedding_model, llm_model, llm_tokenizer, device)
    print(f"Generated Answer: {answer}")
    
    # --- Test Milestone 4: Quiz Generation ---
    print(f"\n--- Testing Quiz Generation ---")
    # We'll use the first chunk of text as the context for the quiz
    quiz_context = text_chunks[0]
    generated_quiz = generate_quiz(quiz_context, llm_model, llm_tokenizer, device)
    print("Generated Quiz:")
    print(generated_quiz)

    # --- Test Explanation Generation ---
    print(f"\n--- Testing Explanation Generation ---")
    
    # We will simulate a quiz question and answer from our context
    simulated_question = "What was the precursor to the Internet mentioned in the text?"
    simulated_answer = "ARPANET (Advanced Research Projects Agency Network)"
    
    explanation = generate_explanation(
        context=quiz_context,
        question=simulated_question,
        correct_answer=simulated_answer,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        device=device
    )
    
    print(f"Question: {simulated_question}")
    print(f"Answer: {simulated_answer}")
    print(f"Generated Explanation: {explanation}")

