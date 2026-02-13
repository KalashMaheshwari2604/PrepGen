import torch
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import numpy as np
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

def segment_into_topics_semantic(text: str, model: SentenceTransformer, threshold: float = 0.3) -> list[str]:
    """Segments a document into topics based on semantic similarity of sentences."""
    sentences = sent_tokenize(text)
    if len(sentences) < 2: return [text]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[:-1], embeddings[1:])
    break_points = np.where(similarities.diag().cpu() < threshold)[0]
    topic_segments = []
    start_index = 0
    for point in break_points:
        end_index = point + 1
        segment = " ".join(sentences[start_index:end_index])
        if segment.strip(): topic_segments.append(segment)
        start_index = end_index
    final_segment = " ".join(sentences[start_index:])
    if final_segment.strip(): topic_segments.append(final_segment)
    return topic_segments

def generate_topic_name(text_chunk: str, model, tokenizer, device: str) -> str:
    """Uses a generative model to create a short title for a text chunk."""
    prompt = f"Generate a short, descriptive title (3-5 words) for the following text:\n\n'{text_chunk}'"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return title