import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from processing import extract_text
import os
import logging

# Setup logger
logger = logging.getLogger("prepgen.summarization")

# --- Import the centrally loaded models ---
# This allows the script to access models loaded by ai_models.py
try:
    from ai_models import models
except ImportError:
    # This is a fallback for static analysis or environments where ai_models.py isn't present
    models = None

# -------------------------------
# Main summarization function
# -------------------------------
def summarize_with_custom_model(input_text: str, 
                                extractor_path: str = "./my_academic_summarizer_scientific",
                                use_direct_llm: bool = False,
                                domain: str = "scientific") -> str:
    """
    Intelligent summarization pipeline that chooses the best strategy based on document length:
    - Short docs (<500 words): T5 Academic Summarizer + LLM polish
    - Medium docs (500-3000 words): Direct LLM for complete coverage
    - Long docs (>3000 words): T5 hierarchical chunking + LLM polish
    
    Args:
        input_text: The text to summarize
        extractor_path: Path to the Academic model (default: ./my_academic_summarizer_scientific)
        use_direct_llm: If True, force direct LLM usage regardless of length
        domain: Content domain - "scientific" (default), "booksum", or "wikihow"
    """
    if not models or not models.academic_summarizer:
         return "Error: Academic summarizer model is not loaded."

    device = models.device
    word_count = len(input_text.split())
    llm = models.llm
    
    if not llm:
        logger.error("Llama model is not loaded")
        return "Error: Llama model is not loaded."
    
    logger.info("Document stats: %d words, %d characters", word_count, len(input_text))
    
    # OPTIMIZED STRATEGY: For documents >3000 words, use direct LLM with intelligent sampling
    # This is MUCH faster than T5 hierarchical chunking on CPU
    if word_count > 3000 or use_direct_llm:
        logger.info("Large document detected - using optimized direct LLM summarization")
        
        # For very long documents, intelligently sample content instead of processing all
        # CRITICAL: Leave room for output tokens! LLM context = input + output
        # Llama 3.2 3B: 4096 tokens total, we want 2048 for output, so ~8000 chars for input
        max_input_chars = 8000  # Reduced to prevent context overflow (was 15000)
        original_length = len(input_text)
        
        if len(input_text) > max_input_chars:
            # Smart sampling: Take beginning, middle, and end sections
            section_size = max_input_chars // 3
            beginning = input_text[:section_size]
            middle_start = len(input_text) // 2 - section_size // 2
            middle = input_text[middle_start:middle_start + section_size]
            end = input_text[-section_size:]
            
            sampled_text = (
                beginning + 
                "\n\n[... content omitted ...]\n\n" + 
                middle + 
                "\n\n[... content omitted ...]\n\n" + 
                end
            )
            logger.info("Sampled document: %d chars from %d original", len(sampled_text), original_length)
            input_text = sampled_text
        else:
            logger.debug("Document fits within limit: %d chars (no sampling needed)", len(input_text))
        
        prompt = f"""[INST]
You are an expert academic content analyst. Analyze this educational presentation/document and create a comprehensive summary.

CRITICAL REQUIREMENTS:
1. Cover EVERY major topic, module, section, and concept mentioned
2. Write 500-700 words to ensure complete coverage of all topics
3. Preserve ALL technical terms, acronyms, product names, and specific details exactly as written
4. Organize with clear section headings for each major topic
5. For each topic: include definitions, key features, components, and important details
6. Use professional academic tone suitable for students
7. Be thorough and comprehensive - mention every topic, even briefly
8. DO NOT add meta-commentary, notes, or apologetic statements
9. End directly after the content - no "Note:", "Summary:", or "Unfortunately" phrases

DOCUMENT:
{input_text}

Create a comprehensive, well-structured summary covering all topics:
[/INST]"""
        
        logger.info("Generating summary with LLM (estimated time: 30-90 seconds)")
        output = llm(prompt, max_tokens=2048, temperature=0.2, top_p=0.9, echo=False)  # Increased for longer summaries
        summary = output['choices'][0]['text'].strip()
        
        # Debug: Check summary length and completion
        word_count_summary = len(summary.split())
        logger.info("Summary generated: %d words, %d characters", word_count_summary, len(summary))
        
        # Check if summary was cut off (ends mid-sentence)
        if not summary.endswith(('.', '!', '?', ')', '"', "'")):
            logger.warning("Summary may be incomplete (doesn't end with proper punctuation)")
            logger.debug("Last 100 chars: ...%s", summary[-100:])
        
        # Remove any unwanted meta-commentary that LLM might add
        meta_phrases = [
            "\n\nNote:", "\nNote:", 
            "\n\nUnfortunately", "\nUnfortunately",
            "\n\nHowever, I", "\nHowever, I",
            "\n\nPlease note", "\nPlease note",
            "\n\nI apologize", "\nI apologize",
            "\n\nThe provided", "\nThe provided"
        ]
        for phrase in meta_phrases:
            if phrase in summary:
                idx = summary.rfind(phrase)
                if idx > 300:  # Only cut if there's substantial content before
                    summary = summary[:idx].strip()
                    break
        
        return summary
    
    # STRATEGY 2: For medium documents (500-3000 words) - direct LLM
    if 500 <= word_count <= 3000:
        logger.info("Medium document - using direct LLM summarization")
        # Truncate if too long for LLM context
        max_input_chars = 12000
        if len(input_text) > max_input_chars:
            input_text = input_text[:max_input_chars] + "..."
        
        prompt = f"""[INST]
You are an expert academic content analyst. Analyze the following educational document and create a comprehensive summary.

CRITICAL REQUIREMENTS:
1. Cover EVERY major topic, module, section, and concept mentioned in the document
2. Write 400-600 words to ensure complete coverage of all topics
3. Preserve ALL technical terms, acronyms, product names, and specific details exactly as written
4. Organize with clear section headings for each major topic
5. For each topic: include definitions, key features, components, and important details
6. Use professional academic tone suitable for students
7. Be thorough and comprehensive - mention every topic, even briefly
8. DO NOT add meta-commentary, notes, or apologetic statements about the content
9. End directly after the content - no "Note:", "Summary:", or "Unfortunately" phrases

DOCUMENT:
{input_text}

Create a comprehensive, well-structured summary covering all topics:
[/INST]"""
        output = llm(prompt, max_tokens=1536, temperature=0.2, top_p=0.9, echo=False)
        summary = output['choices'][0]['text'].strip()
        logger.info("Summary generated successfully")
        
        # Remove any unwanted meta-commentary that LLM might add
        meta_phrases = [
            "\n\nNote:", "\nNote:", 
            "\n\nUnfortunately", "\nUnfortunately",
            "\n\nHowever, I", "\nHowever, I",
            "\n\nPlease note", "\nPlease note",
            "\n\nI apologize", "\nI apologize",
            "\n\nThe provided", "\nThe provided"
        ]
        for phrase in meta_phrases:
            if phrase in summary:
                idx = summary.rfind(phrase)
                if idx > 300:  # Only cut if there's substantial content before
                    summary = summary[:idx].strip()
                    break
        
        return summary
    
    # STRATEGY 3: For short documents (<500 words) - Use T5 for extraction + LLM polish
    # This is the only case where we use T5, since it's fast for short docs
    
    logger.info("Short document - using T5 extraction + LLM polish")
    summarizer = models.academic_summarizer
    
    # Direct T5 summarization for short documents
    key_points = summarizer.summarize(
        text=input_text,
            domain=domain,
            max_input_len=1024,
            max_new_tokens=300,
            num_beams=4
        )
    
    # Polish the extracted key points with LLM
    prompt = f"""[INST]
You are an expert academic content analyst. You have been given key points extracted from an educational document.

Your task: Create a comprehensive summary that covers ALL topics mentioned.

CRITICAL REQUIREMENTS:
1. Cover EVERY topic/module/section mentioned in the key points
2. Write 300-450 words ensuring all topics are included
3. Preserve ALL technical terms, acronyms, and specific details
4. Organize with clear section headings for each major topic
5. For each topic, include: definition, key concepts, and important details
6. Use professional academic tone
7. Be comprehensive and detailed - don't focus on just one topic
8. DO NOT add meta-commentary or notes about your writing process
9. End directly after the content

KEY POINTS EXTRACTED:
{key_points}

Write a comprehensive summary covering all topics:
[/INST]"""
    
    output = llm(prompt, max_tokens=1024, temperature=0.3, top_p=0.9, echo=False)
    polished_summary = output['choices'][0]['text'].strip()
    
    # Remove any meta-commentary
    meta_phrases = [
        "\n\nNote:", "\nNote:", 
        "\n\nUnfortunately", "\nUnfortunately",
        "\n\nHowever, I", "\nHowever, I",
        "\n\nPlease note", "\nPlease note"
    ]
    
    for phrase in meta_phrases:
        if phrase.lower() in polished_summary.lower():
            idx = polished_summary.lower().rfind(phrase.lower())
            if idx > 200:
                polished_summary = polished_summary[:idx].strip()
                break
    
    return polished_summary

# -------------------------------
# Optional: test via CLI
# -------------------------------
if __name__ == "__main__":
    # This block allows you to test this script directly.
    # It will use the models loaded by ai_models.py.
    if not models or not models.llm or not models.custom_summary_model:
        logger.error("One or more models could not be loaded from ai_models.py")
        logger.error("Please ensure ai_models.py is configured correctly before running this test")
    else:
        file_path = input("Enter text file path to summarize: ").strip()
        if os.path.exists(file_path):
            text_content = extract_text(file_path)
            if text_content:
                summary = summarize_with_custom_model(text_content)
                print("\n--- FINAL SUMMARY ---\n")
                print(summary)
            else:
                logger.error("Could not extract text from the file")
        else:
            logger.error("File not found: %s", file_path)
