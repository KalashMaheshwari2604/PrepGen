"""
Academic Summarizer Inference Module

Production-ready inference class for the fine-tuned academic summarizer model.
Supports domain-aware summarization with optimized settings for academic content.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import Optional, List


class AcademicSummarizer:
    """
    Inference wrapper for the academic summarizer model.
    
    Trained on mixed dataset:
    - 70% Scientific papers (arXiv)
    - 20% BookSum (long documents)
    - 10% WikiHow (instructional content)
    
    Supports domain-aware prompting for different content types.
    """
    
    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Initialize the academic summarizer.
        
        Args:
            model_dir: Path to the fine-tuned model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ Academic Summarizer loaded on {self.device}")
    
    @torch.inference_mode()
    def summarize(
        self,
        text: str,
        domain: str = "scientific",
        max_input_len: int = 640,
        max_new_tokens: int = 160,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> str:
        """
        Generate a summary for the input text.
        
        Args:
            text: Input text to summarize
            domain: Content domain - one of:
                - "scientific": Academic papers, technical content
                - "booksum": Long-form documents, book chapters
                - "wikihow": Instructional content, guidelines
            max_input_len: Maximum input sequence length (default: 640)
            max_new_tokens: Maximum tokens to generate (default: 160)
            num_beams: Number of beams for beam search (default: 4)
            length_penalty: Length penalty for generation (default: 1.0)
            early_stopping: Whether to stop early (default: True)
        
        Returns:
            Generated summary as string
        """
        # Map domain to prefix
        prefix_map = {
            "scientific": "summarize scientific paper: ",
            "booksum": "summarize book chapter: ",
            "wikihow": "summarize instructions: ",
        }
        
        if domain not in prefix_map:
            print(f"⚠️  Warning: Unknown domain '{domain}', using 'scientific'")
            domain = "scientific"
        
        prefix = prefix_map[domain]
        prompt = prefix + text
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len
        ).to(self.device)
        
        # Generate summary
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping
        )
        
        # Decode and return
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary
    
    def batch_summarize(
        self,
        texts: List[str],
        domain: str = "scientific",
        max_input_len: int = 640,
        max_new_tokens: int = 160,
        num_beams: int = 4
    ) -> List[str]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts
            domain: Content domain (same for all texts)
            max_input_len: Maximum input sequence length
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
        
        Returns:
            List of generated summaries
        """
        return [
            self.summarize(
                text, 
                domain=domain,
                max_input_len=max_input_len,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )
            for text in texts
        ]
    
    def chunk_and_summarize(
        self,
        text: str,
        domain: str = "scientific",
        chunk_size: int = 800,
        overlap: int = 200,
        hierarchical: bool = True
    ) -> str:
        """
        Handle long documents by chunking and optional hierarchical summarization.
        
        Args:
            text: Long input text
            domain: Content domain
            chunk_size: Token size for each chunk
            overlap: Overlap tokens between chunks
            hierarchical: If True, summarize chunks then summarize summaries
        
        Returns:
            Final summary
        """
        # Tokenize full text
        tokens = self.tokenizer.encode(text)
        
        # Check if chunking is needed
        if len(tokens) <= 640:
            return self.summarize(text, domain=domain)
        
        # Create chunks with overlap
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            start += (chunk_size - overlap)
        
        print(f"📄 Split into {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = self.batch_summarize(chunks, domain=domain)
        
        if not hierarchical:
            # Just concatenate chunk summaries
            return " ".join(chunk_summaries)
        
        # Hierarchical: summarize the summaries
        combined = " ".join(chunk_summaries)
        print(f"🔄 Running hierarchical summarization...")
        final_summary = self.summarize(combined, domain=domain, max_new_tokens=256)
        
        return final_summary


# Example usage
if __name__ == "__main__":
    # Initialize
    summarizer = AcademicSummarizer("../my_academic_summarizer_scientific")
    
    # Test text
    test_text = """
    Cloud Computing (CC-702IT0C026) is a comprehensive course designed for B.Tech and MBA students 
    in their seventh semester. The course covers cloud service models (IaaS, PaaS, SaaS), 
    deployment models (public, private, hybrid), virtualization technologies, and cloud security.
    """
    
    # Generate summary
    summary = summarizer.summarize(test_text, domain="scientific")
    print(f"\n📝 Summary: {summary}")
    
    # Test with different domain
    instructions = """
    To install Python packages, open your terminal and use pip. 
    First, update pip with 'pip install --upgrade pip'. 
    Then install packages with 'pip install package-name'.
    """
    
    summary2 = summarizer.summarize(instructions, domain="wikihow")
    print(f"\n📝 Instructions Summary: {summary2}")
