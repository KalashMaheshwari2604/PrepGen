import os
from llama_cpp import Llama
import json
from typing import List, Dict, Any, Tuple
from processing import extract_text
import logging

# Setup logger
logger = logging.getLogger("prepgen.quiz_rag")

# Optional: SerpAPI for web search fallback
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False
    logger.warning("SerpAPI not installed. Web search fallback will be disabled")

try:
    from ai_models import models
except ImportError:
    models = None

# --- Helper function to prevent context overflow ---
def truncate_text(text: str, max_chars: int = 20000) -> str:
    """Limits text to a max number of characters to prevent model context overflow."""
    if len(text) > max_chars:
        logger.warning("Context truncated from %d to %d characters", len(text), max_chars)
        return text[:max_chars]
    return text


def sample_text_evenly(text: str, max_chars: int = 18000) -> str:
    """
    Sample text from beginning, middle, and end to cover all topics.
    This ensures questions cover the entire document, not just the start.
    """
    if len(text) <= max_chars:
        return text
    
    # Take samples from 3 parts: beginning (40%), middle (30%), end (30%)
    chunk_size = max_chars // 3
    
    beginning = text[:int(chunk_size * 1.3)]
    
    mid_start = len(text) // 2 - chunk_size // 2
    middle = text[mid_start:mid_start + chunk_size]
    
    end = text[-chunk_size:]
    
    sampled = f"{beginning}\n\n[... middle content ...]\n\n{middle}\n\n[... later content ...]\n\n{end}"
    
    logger.info("Sampled text from beginning, middle, and end to cover all topics")
    return sampled


def extract_topics(text: str) -> str:
    """
    Extract main topics/sections from the document to guide balanced question generation.
    """
    lines = text.split('\n')
    topics = []
    
    for line in lines:
        line = line.strip()
        # Identify potential topic headers (short lines, title case, or all caps)
        if line and len(line) < 100:
            if line.isupper() or (line[0].isupper() and ':' in line) or line.endswith(':'):
                topics.append(line.rstrip(':'))
    
    # If we found topics, return them
    if topics and len(topics) >= 3:
        unique_topics = list(dict.fromkeys(topics))[:15]  # Max 15 topics
        return "\n".join(f"- {t}" for t in unique_topics)
    
    # Fallback: extract first sentence of each paragraph as potential topics
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    topic_sentences = []
    for para in paragraphs[:15]:
        first_sentence = para.split('.')[0].strip()
        if 10 < len(first_sentence) < 150:
            topic_sentences.append(first_sentence)
    
    return "\n".join(f"- {t}" for t in topic_sentences[:10]) if topic_sentences else "Multiple topics covered"


# --- 1. Web Search Function ---
def search_web_with_serpapi(question: str) -> str:
    """
    Performs a web search using SerpAPI if the answer isn't in the document.
    """
    if not SERPAPI_AVAILABLE:
        return "Web search is not available. Install serpapi with: pip install google-search-results"
    
    logger.info("Performing web search fallback")
    try:
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_key:
            return "Web search is not configured. Please set the SERPAPI_API_KEY environment variable."
        params = {"api_key": serpapi_key, "q": question, "engine": "google"}
        search = GoogleSearch(params)
        results = search.get_dict()
        if "answer_box" in results and "snippet" in results["answer_box"]:
            return results["answer_box"]["snippet"]
        elif "organic_results" in results and "snippet" in results["organic_results"][0]:
            return results["organic_results"][0]["snippet"]
        else:
            return "Could not find a direct answer on the web."
    except Exception as e:
        logger.error("Error during web search: %s", e)
        return "An error occurred while searching the web."

# --- 2. RAG Chatbot with Intelligent Context Matching ---
def generate_rag_answer(question: str, context: str, llm: Llama, conversation_history: List[Dict] = None) -> Tuple[str, str]:
    """
    Generates an answer using RAG with improved context matching.
    Includes conversation history for context awareness.
    Returns the answer and its source ('document' or 'web').
    """
    if not llm:
        return "Error: LLM model is not available.", "error"
    
    # Use more context for better answers
    safe_context = sample_text_evenly(context, max_chars=18000)
    
    # Build conversation context if available
    conversation_context = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 exchanges
        for exchange in recent_history:
            conversation_context += f"User: {exchange.get('question', '')}\nAssistant: {exchange.get('answer', '')}\n\n"
    
    # Step 1: Try to answer from document with better prompt
    answer_prompt = f"""[INST]
You are a helpful AI assistant answering questions about a document.

{"Previous conversation:\n" + conversation_context if conversation_context else ""}

Document content:
{safe_context}

Question: {question}

Instructions:
1. If the answer is in the document above, provide a clear and complete answer
2. Use information ONLY from the document
3. If you're not sure or the information is not in the document, say exactly: "I cannot find this information in the document."
4. Be specific and cite relevant details from the document
5. Keep answers concise but complete (2-4 sentences)

Answer:
[/INST]"""
    
    try:
        answer_output = llm(
            answer_prompt, 
            max_tokens=600,
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.2,
            echo=False
        )
        answer = answer_output['choices'][0]['text'].strip()
        
        # Check if model explicitly said it can't answer
        cant_answer_phrases = [
            "cannot find this information",
            "not found in the document",
            "not mentioned in the document",
            "document does not contain",
            "information is not available",
            "not provided in the document"
        ]
        
        answer_lower = answer.lower()
        
        # If model explicitly says it can't answer, fall back to web search
        if any(phrase in answer_lower for phrase in cant_answer_phrases):
            logger.info("Model could not find answer in document. Attempting web search fallback")
            web_answer = search_web_with_serpapi(question)
            final_answer = f"{web_answer}\n\n(Note: This information was not found in your document and came from a web search.)"
            return final_answer, "web"
        
        # If answer is too short (< 10 words), might be hallucination
        if len(answer.split()) < 10:
            logger.info("Answer too short, trying web search")
            web_answer = search_web_with_serpapi(question)
            final_answer = f"{web_answer}\n\n(Note: This information was not found in your document and came from a web search.)"
            return final_answer, "web"
        
        # Otherwise, we have a good answer from the document
        logger.info("Answer generated from document context")
        return answer, "document"
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error: Could not generate answer.", "error"

# --- 3. Quiz Generation with Explanations (FIXED) ---
def generate_quiz_questions(context: str, num_questions: int, llm: Llama, previously_asked: List[str] = []) -> List[Dict[str, Any]]:
    """
    Generates unique quiz questions with options, correct answer, and an explanation.
    It avoids generating questions that have already been asked.
    Uses batch generation to prevent JSON cutoff issues.
    """
    if not llm:
        return [{"error": "LLM model is not available."}]

    # If requesting many questions, generate in batches
    BATCH_SIZE = 3  # Generate 3 questions at a time for better JSON reliability
    
    if num_questions > BATCH_SIZE:
        logger.info(f"Generating {num_questions} questions in batches of {BATCH_SIZE}")
        all_questions = []
        remaining = num_questions
        
        while remaining > 0 and len(all_questions) < num_questions:
            batch_size = min(BATCH_SIZE, remaining)
            logger.info(f"Generating batch: {batch_size} questions (total so far: {len(all_questions)})")
            
            # Generate batch
            batch_questions = _generate_quiz_batch(
                context, 
                batch_size, 
                llm, 
                previously_asked + [q["question"] for q in all_questions]
            )
            
            if batch_questions and not ("error" in batch_questions[0]):
                all_questions.extend(batch_questions)
                remaining -= len(batch_questions)
                logger.info(f"Batch successful: {len(batch_questions)} questions added")
            else:
                logger.warning(f"Batch generation failed, stopping early with {len(all_questions)} questions")
                break
        
        logger.info(f"Total questions generated: {len(all_questions)}/{num_questions}")
        return all_questions if all_questions else [{"error": "Failed to generate questions"}]
    else:
        # For small requests, generate directly
        return _generate_quiz_batch(context, num_questions, llm, previously_asked)


def _generate_quiz_batch(context: str, num_questions: int, llm: Llama, previously_asked: List[str] = []) -> List[Dict[str, Any]]:
    """
    Generate quiz questions using SIMPLE text format (not JSON).
    This is much more reliable for Llama 3.2 3B.
    """
    # Sample text to fit context
    safe_context = sample_text_evenly(context, max_chars=5000)
    
    prompt = f"""[INST]
Create {num_questions} multiple-choice quiz questions from this document.

Use EXACTLY this format:

Q1: What is the main concept?
A) First option
B) Second option
C) Third option
D) Fourth option
ANSWER: A
EXPLAIN: Why this is correct

Q2: Another question about the content?
A) Option one
B) Option two
C) Option three
D) Option four
ANSWER: C
EXPLAIN: Brief explanation

Document content:
{safe_context}

Generate {num_questions} questions now using the format above:
[/INST]"""

    logger.info(f"Generating {num_questions} questions in simple text format")
    
    for attempt in range(2):  # 2 attempts
        try:
            output = llm(
                prompt,
                max_tokens=2500,
                temperature=0.5,
                repeat_penalty=1.2,
                echo=False
            )
            
            text = output['choices'][0]['text'].strip()
            logger.info(f"Generated {len(text)} chars, parsing simple format...")
            
            # Parse simple text format
            questions = []
            current_q = {}
            opts = []
            
            for line in text.split('\n'):
                line = line.strip()
                
                # New question: Q1:, Q2:, etc.
                if line.startswith('Q') and ':' in line[:5]:
                    # Save previous question if complete
                    if current_q and 'question' in current_q and len(current_q.get('options', [])) == 4:
                        questions.append(current_q)
                    
                    # Start new question
                    current_q = {'question': line.split(':', 1)[1].strip()}
                    opts = []
                
                # Options: A), B), C), D)
                elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                    opt_text = line[2:].strip()
                    if opt_text:
                        opts.append(opt_text)
                        current_q['options'] = opts.copy()
                
                # Answer: ANSWER: A
                elif line.startswith('ANSWER:'):
                    answer_letter = line[7:].strip()
                    if answer_letter and answer_letter[0].upper() in 'ABCD' and 'options' in current_q:
                        idx = ord(answer_letter[0].upper()) - ord('A')
                        if idx < len(current_q['options']):
                            current_q['correct_answer'] = current_q['options'][idx]
                
                # Explanation: EXPLAIN:
                elif line.startswith('EXPLAIN:'):
                    explain_text = line[8:].strip()
                    if explain_text:
                        current_q['explanation'] = explain_text
                        current_q['difficulty'] = 'easy'
            
            # Don't forget last question
            if current_q and 'question' in current_q and len(current_q.get('options', [])) == 4:
                questions.append(current_q)
            
            # Validate questions
            valid_questions = []
            for q in questions:
                if all(k in q for k in ['question', 'options', 'correct_answer', 'explanation']):
                    if isinstance(q['options'], list) and len(q['options']) == 4:
                        valid_questions.append(q)
            
            if valid_questions:
                logger.info(f"Successfully generated {len(valid_questions)} valid questions")
                return valid_questions
            else:
                logger.warning(f"Attempt {attempt + 1}: No valid questions parsed")
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            continue

    # All attempts failed
    logger.error("Failed to generate valid quiz questions after all attempts")
    if not previously_asked:
        return [{
            "question": "Quiz generation temporarily unavailable",
            "options": [
                "The system is processing your document", 
                "Please try again in a moment", 
                "Content may be too complex",
                "Check document format"
            ],
            "correct_answer": "Please try again in a moment",
            "explanation": "The quiz generator encountered difficulties. Please refresh and try again.",
            "difficulty": "easy"
        }]
    else:
        return [{
            "question": "Quiz Complete! 🎉",
            "options": [
                "You've answered all available questions", 
                "Great job!", 
                "Review your answers",
                "Try another document"
            ],
            "correct_answer": "You've answered all available questions",
            "explanation": "Congratulations! You have successfully completed the quiz for this document.",
            "difficulty": "easy"
        }]

