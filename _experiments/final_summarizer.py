import torch
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# --- IMPORT YOUR CUSTOM MODULES ---
from processing import extract_text
from topic_pipeline import segment_into_topics_semantic, generate_topic_name
from ai_pipeline import extractive_summary 

# --- Main Pipeline ---
if __name__ == "__main__":
    # 1. Load Necessary Models
    print("--- Loading Models ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    title_model_name = "google/flan-t5-base"
    title_model = T5ForConditionalGeneration.from_pretrained(title_model_name).to(device)
    title_tokenizer = T5Tokenizer.from_pretrained(title_model_name)
    
    print("Models loaded successfully.\n")

    # 2. Load and Segment the Document
    document_text = extract_text('sample.txt')
    print("--- Segmenting document into topics ---")
    # Call the imported function
    topics = segment_into_topics_semantic(document_text, embedding_model, threshold=0.3)
    print(f"Found {len(topics)} distinct topics.\n")
    
    # 3. Summarize Each Topic and Compile the Final Result
    summary_percent = 0.5
    
    final_summary_parts = []

    for i, topic_text in enumerate(topics):
        print(f"--- Processing Topic {i+1}/{len(topics)} ---")
        
        print("Generating topic name...")
        # Call the imported function
        topic_name = generate_topic_name(topic_text, title_model, title_tokenizer, device)
        
        print("Generating extractive summary...")
        # Call the imported extractive_summary function from ai_pipeline
        topic_summary = extractive_summary(topic_text, summary_percentage=summary_percent)
        
        final_summary_parts.append(f"## {topic_name}\n{topic_summary}\n")

    # 4. Print the Final Structured Summary
    print("\n" + "="*80 + "\n")
    print("--- FINAL STRUCTURED SUMMARY ---")
    final_document_summary = "\n".join(final_summary_parts)
    print(final_document_summary)
    print("\n" + "="*80 + "\n")
