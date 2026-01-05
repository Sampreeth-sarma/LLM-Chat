from abc import ABC, abstractmethod
import time

class BaseModelProvider(ABC):
    """Abstract base class for all model providers"""
    
    def __init__(self):
        self.metrics = {
            "generation_speed": [],  # tokens/second
            "latency": [],           # time to first token
            "total_tokens": 0,
            "total_time": 0
        }
        self.use_kv_cache = True
    
    @abstractmethod
    def generate(self, messages, **kwargs):
        """Generate text from messages"""
        pass
    
    @classmethod
    @abstractmethod
    def list_models(cls):
        """List available models"""
        pass
    
    def set_kv_cache(self, enabled=True):
        """Enable or disable KV cache"""
        self.use_kv_cache = enabled
    
    def _track_generation_metrics(self, start_time, tokens_generated):
        """Track generation metrics"""
        generation_time = time.time() - start_time
        
        if generation_time > 0 and tokens_generated > 0:
            tokens_per_second = tokens_generated / generation_time
            self.metrics["generation_speed"].append(tokens_per_second)
            self.metrics["total_tokens"] += tokens_generated
            self.metrics["total_time"] += generation_time
            
            # Keep only last 10 measurements for rolling average
            if len(self.metrics["generation_speed"]) > 10:
                self.metrics["generation_speed"] = self.metrics["generation_speed"][-10:]
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        if not self.metrics["generation_speed"]:
            return {"avg_speed": 0, "total_tokens": 0, "total_time": 0}
        
        return {
            "avg_speed": sum(self.metrics["generation_speed"]) / len(self.metrics["generation_speed"]),
            "total_tokens": self.metrics["total_tokens"],
            "total_time": self.metrics["total_time"]
        }
    
    def close(self):
        """Release resources (GPU memory, processes) if applicable."""
        return
