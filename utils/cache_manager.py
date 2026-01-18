import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
import logging

class CacheManager:
    """
    File-based cache manager for graphs and LLM responses
    """
    
    def __init__(self, cache_dir="cache", ttl_hours=24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.graphs_dir = self.cache_dir / "graphs"
        self.llm_dir = self.cache_dir / "llm_responses"
        
        self.graphs_dir.mkdir(exist_ok=True)
        self.llm_dir.mkdir(exist_ok=True)
        
        self.ttl = timedelta(hours=ttl_hours)
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_expired(self, filepath: Path) -> bool:
        """Check if cache file is expired"""
        if not filepath.exists():
            return True
        
        modified_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        return datetime.now() - modified_time > self.ttl
    
    def cache_graph(self, graph_data: dict, graph_path: str) -> None:
        """Cache graph metadata"""
        key = self._generate_key(graph_data)
        cache_file = self.graphs_dir / f"{key}.json"
        
        cache_entry = {
            "metadata": graph_data,
            "path": graph_path,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_entry, f, indent=2)
        
        self.logger.info(f"Cached graph: {graph_path}")
    
    def get_cached_graph(self, graph_data: dict) -> Optional[str]:
        """Retrieve cached graph path if exists"""
        key = self._generate_key(graph_data)
        cache_file = self.graphs_dir / f"{key}.json"
        
        if self._is_expired(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_entry = json.load(f)
            
            graph_path = Path(cache_entry["path"])
            if graph_path.exists():
                self.logger.info(f"Cache hit for graph: {graph_path}")
                return str(graph_path)
        except Exception as e:
            self.logger.error(f"Cache read error: {e}")
        
        return None
    
    def cache_llm_response(self, prompt: str, response: str, 
                          image_path: Optional[str] = None) -> None:
        """Cache LLM response"""
        cache_data = {
            "prompt": prompt,
            "image_path": image_path
        }
        
        key = self._generate_key(cache_data)
        cache_file = self.llm_dir / f"{key}.pkl"
        
        cache_entry = {
            "prompt": prompt,
            "response": response,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_entry, f)
        
        self.logger.info(f"Cached LLM response for prompt hash: {key[:8]}")
    
    def get_cached_llm_response(self, prompt: str, 
                                image_path: Optional[str] = None) -> Optional[str]:
        """Retrieve cached LLM response"""
        cache_data = {
            "prompt": prompt,
            "image_path": image_path
        }
        
        key = self._generate_key(cache_data)
        cache_file = self.llm_dir / f"{key}.pkl"
        
        if self._is_expired(cache_file):
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_entry = pickle.load(f)
            
            self.logger.info(f"Cache hit for LLM response: {key[:8]}")
            return cache_entry["response"]
        except Exception as e:
            self.logger.error(f"LLM cache read error: {e}")
        
        return None
    
    def clear_cache(self, cache_type: str = "all") -> None:
        """Clear cache files"""
        if cache_type in ["graphs", "all"]:
            for file in self.graphs_dir.glob("*.json"):
                file.unlink()
            self.logger.info("Cleared graph cache")
        
        if cache_type in ["llm", "all"]:
            for file in self.llm_dir.glob("*.pkl"):
                file.unlink()
            self.logger.info("Cleared LLM cache")
    
    def clear_expired(self) -> None:
        """Remove expired cache entries"""
        count = 0
        
        for file in self.graphs_dir.glob("*.json"):
            if self._is_expired(file):
                file.unlink()
                count += 1
        
        for file in self.llm_dir.glob("*.pkl"):
            if self._is_expired(file):
                file.unlink()
                count += 1
        
        self.logger.info(f"Cleared {count} expired cache entries")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        graph_files = list(self.graphs_dir.glob("*.json"))
        llm_files = list(self.llm_dir.glob("*.pkl"))
        
        graph_size = sum(f.stat().st_size for f in graph_files)
        llm_size = sum(f.stat().st_size for f in llm_files)
        
        return {
            "graph_entries": len(graph_files),
            "llm_entries": len(llm_files),
            "graph_size_mb": graph_size / (1024 * 1024),
            "llm_size_mb": llm_size / (1024 * 1024),
            "total_size_mb": (graph_size + llm_size) / (1024 * 1024)
        }
