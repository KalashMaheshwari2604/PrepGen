import torch
from transformers import LEDForConditionalGeneration, LEDTokenizer
from processing import extract_text, chunk_text

def get_led_summary(text_to_summarize: str, model, tokenizer, device: str) -> str:
    """Helper function to generate a summary using the LED model."""
    
    inputs = tokenizer([text_to_summarize], 
                       max_length=1024, # LED can handle much longer, but we'll keep this for chunking
                       return_tensors="pt",
                       truncation=True).to(device)
                       
    outputs = model.generate(
        inputs["input_ids"], 
        max_new_tokens=512,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# --- Main Test Block ---
if __name__ == "__main__":
    # 1. Load the single expert summarization model
    print("--- Loading the LED Summarization Model ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_name = "pszemraj/led-large-book-summary"
    model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    
    print("Model loaded successfully.\n")

    # 2. Load and chunk the document
    print("--- Processing Document ---")
    text = extract_text('sample.txt') 
    # Using a smaller chunk size to create more, smaller chunks for this long text
    chunks = chunk_text(text, chunk_size=1024, chunk_overlap=100)
    print(f"Document split into {len(chunks)} chunks.\n")
    
    # --- 3. Full Map-Reduce Summarization ---
    
    # Map step: Summarize each chunk
    print("--- Generating Summaries for Each Chunk (Map Step) ---")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        chunk_summary = get_led_summary(chunk, model, tokenizer, device)
        chunk_summaries.append(chunk_summary)

    # Recursive Reduce step
    # We'll keep summarizing the summaries until we have only one left.
    current_summaries = chunk_summaries
    while len(current_summaries) > 1:
        print(f"\n--- Reducing {len(current_summaries)} summaries into a smaller set ---")
        new_summaries = []
        # Group the summaries in batches of 5
        for i in range(0, len(current_summaries), 5):
            batch = current_summaries[i:i+5]
            combined_batch_text = "\n".join(batch)
            
            print(f"Summarizing batch {i//5 + 1}...")
            batch_summary = get_led_summary(combined_batch_text, model, tokenizer, device)
            new_summaries.append(batch_summary)
        
        current_summaries = new_summaries

    final_summary = current_summaries[0]

    print("\n" + "="*50 + "\n")
    print("--- FINAL SUMMARY ---")
    print(final_summary)
    print("\n" + "="*50 + "\n")