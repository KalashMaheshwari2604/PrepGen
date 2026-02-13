"""
Caching system for embeddings and FAISS indices
Reduces computation time for repeated documents
"""
import hashlib
import pickle
import os
from typing import Optional, Tuple
import numpy as np
import faiss
from datetime import datetime, timedelta


class EmbeddingCache:
    """Cache for document embeddings and FAISS indices"""
    
    def __init__(self, cache_dir: str = "./cache", max_age_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_age = timedelta(hours=max_age_hours)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "indices"), exist_ok=True)
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a unique hash for document text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, filepath: str) -> bool:
        """Check if cache file is not expired"""
        if not os.path.exists(filepath):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        return datetime.now() - file_time < self.max_age
    
    def get_embeddings(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embeddings for text.
        
        Args:
            text: Document text
            
        Returns:
            Cached embeddings array or None if not found/expired
        """
        text_hash = self._get_text_hash(text)
        cache_file = os.path.join(self.cache_dir, "embeddings", f"{text_hash}.npy")
        
        if self._is_cache_valid(cache_file):
            try:
                embeddings = np.load(cache_file)
                print(f"Loaded embeddings from cache ({text_hash[:8]}...)")
                return embeddings
            except Exception as e:
                print(f" Failed to load cached embeddings: {e}")
                return None
        
        return None
    
    def save_embeddings(self, text: str, embeddings: np.ndarray) -> None:
        """
        Save embeddings to cache.
        
        Args:
            text: Document text
            embeddings: Numpy array of embeddings
        """
        text_hash = self._get_text_hash(text)
        cache_file = os.path.join(self.cache_dir, "embeddings", f"{text_hash}.npy")
        
        try:
            np.save(cache_file, embeddings)
            print(f" Saved embeddings to cache ({text_hash[:8]}...)")
        except Exception as e:
            print(f" Failed to save embeddings to cache: {e}")
    
    def get_faiss_index(self, text: str) -> Optional[Tuple[faiss.Index, list]]:
        """
        Get cached FAISS index and chunks.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (FAISS index, text chunks) or None if not found/expired
        """
        text_hash = self._get_text_hash(text)
        index_file = os.path.join(self.cache_dir, "indices", f"{text_hash}.index")
        chunks_file = os.path.join(self.cache_dir, "indices", f"{text_hash}.pkl")
        
        if self._is_cache_valid(index_file) and self._is_cache_valid(chunks_file):
            try:
                # Load FAISS index
                index = faiss.read_index(index_file)
                
                # Load chunks
                with open(chunks_file, 'rb') as f:
                    chunks = pickle.load(f)
                
                print(f" Loaded FAISS index from cache ({text_hash[:8]}...)")
                return index, chunks
            except Exception as e:
                print(f" Failed to load cached FAISS index: {e}")
                return None
        
        return None
    
    def save_faiss_index(self, text: str, index: faiss.Index, chunks: list) -> None:
        """
        Save FAISS index and chunks to cache.
        
        Args:
            text: Document text
            index: FAISS index
            chunks: List of text chunks
        """
        text_hash = self._get_text_hash(text)
        index_file = os.path.join(self.cache_dir, "indices", f"{text_hash}.index")
        chunks_file = os.path.join(self.cache_dir, "indices", f"{text_hash}.pkl")
        
        try:
            # Save FAISS index
            faiss.write_index(index, index_file)
            
            # Save chunks
            with open(chunks_file, 'wb') as f:
                pickle.dump(chunks, f)
            
            print(f" Saved FAISS index to cache ({text_hash[:8]}...)")
        except Exception as e:
            print(f" Failed to save FAISS index to cache: {e}")
    
    def clear_cache(self) -> int:
        """
        Clear all cache files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for subdir in ["embeddings", "indices"]:
            dir_path = os.path.join(self.cache_dir, subdir)
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                try:
                    os.remove(filepath)
                    count += 1
                except Exception as e:
                    print(f"⚠️ Failed to delete {filepath}: {e}")
        
        print(f"🧹 Cleared {count} cache files")
        return count
    
    def clear_expired_cache(self) -> int:
        """
        Clear only expired cache files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for subdir in ["embeddings", "indices"]:
            dir_path = os.path.join(self.cache_dir, subdir)
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                if not self._is_cache_valid(filepath):
                    try:
                        os.remove(filepath)
                        count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to delete {filepath}: {e}")
        
        if count > 0:
            print(f"🧹 Cleared {count} expired cache files")
        return count
    
    def get_cache_stats(self) -> dict:
        """Get statistics about cache usage"""
        stats = {
            "embeddings": {"count": 0, "total_size_mb": 0},
            "indices": {"count": 0, "total_size_mb": 0}
        }
        
        for subdir in ["embeddings", "indices"]:
            dir_path = os.path.join(self.cache_dir, subdir)
            total_size = 0
            count = 0
            
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
                    count += 1
            
            stats[subdir]["count"] = count
            stats[subdir]["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats


# Global cache instance
embedding_cache = EmbeddingCache(cache_dir="./cache", max_age_hours=24)
