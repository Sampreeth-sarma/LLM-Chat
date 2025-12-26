import requests
import json
import time
from .base import BaseModelProvider

class OllamaProvider(BaseModelProvider):
    def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url
        self.model_family = self._detect_model_family(model_name)
        
    def generate(self, messages, temperature=0.7, max_tokens=2048, 
                top_p=0.9, top_k=40, stop=None, **kwargs):
        """Generate response using Ollama API with streaming"""
        prompt = self._format_messages(messages)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "stop": stop,
                "options": {
                    "num_ctx": kwargs.get("context_size", 4096),
                    "use_kv_cache": self.use_kv_cache,  # KV-cache toggle
                    **kwargs.get("options", {})
                }
            },
            stream=True
        )
        
        start_time = time.time()
        first_token = True
        token_count = 0
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    response_chunk = data.get('response', '')
                    
                    if first_token and response_chunk.strip():
                        first_token = False
                        self.metrics["latency"].append(time.time() - start_time)
                        # Keep only last 10 latencies
                        if len(self.metrics["latency"]) > 10:
                            self.metrics["latency"] = self.metrics["latency"][-10:]
                    
                    token_count += 1
                    yield response_chunk
                    
                    if data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        # Track metrics after generation completes
        self._track_generation_metrics(start_time, token_count)
    
    def list_models(self):
        """List available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except:
            pass
        # Fallback list of common models
        return ["llama2", "llama2:13b", "llama3", "mistral", "mistral:instruct", 
                "codellama", "qwen:14b", "qwen:72b", "deepseek-coder", "phi3"]
                
    def get_model_info(self, model_name=None):
        """Get detailed information about a specific model"""
        if model_name is None:
            model_name = self.model_name
            
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"name": model_name, "details": "Information not available"}
    
    def _format_messages(self, messages):
        """Format messages into a prompt string based on model family"""
        # Get appropriate template for the model
        model_family = self._detect_model_family(self.model_name)
        
        if model_family == "llama":
            return self._format_llama_messages(messages)
        elif model_family == "mistral":
            return self._format_mistral_messages(messages)
        elif model_family == "qwen":
            return self._format_qwen_messages(messages)
        elif model_family == "deepseek":
            return self._format_deepseek_messages(messages)
        else:
            # Default formatting
            return self._format_generic_messages(messages)
    
    def _format_llama_messages(self, messages):
        """Format messages for Llama models"""
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
                
        formatted += "<|assistant|>\n"
        return formatted
    
    def _format_mistral_messages(self, messages):
        """Format messages for Mistral models"""
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"[INST] {content} [/INST]\n"
            elif role == "user":
                formatted += f"[INST] {content} [/INST]\n"
            elif role == "assistant":
                formatted += f"{content}\n"
                
        return formatted
    
    def _format_qwen_messages(self, messages):
        """Format messages for Qwen models"""
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    def _format_deepseek_messages(self, messages):
        """Format messages for DeepSeek models"""
        formatted = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted += f"<｜system｜>\n{content}\n"
            elif role == "user":
                formatted += f"<｜user｜>\n{content}\n"
            elif role == "assistant":
                formatted += f"<｜assistant｜>\n{content}\n"
                
        formatted += "<｜assistant｜>\n"
        return formatted
    
    def _format_generic_messages(self, messages):
        """Generic message formatting fallback"""
        formatted = ""
        system_content = None
        
        # Extract system message if present
        for message in messages:
            if message.get("role") == "system":
                system_content = message.get("content")
                break
        
        # If system message exists, prepend it
        if system_content:
            formatted += f"{system_content}\n\n"
        
        # Add conversation history
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "system":
                continue  # Already handled
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        formatted += "Assistant: "
        return formatted
    
    def _detect_model_family(self, model_name):
        """Detect the model family from the model name"""
        model_name = model_name.lower()
        
        if any(name in model_name for name in ["llama", "llama2", "llama3"]):
            return "llama"
        elif "mistral" in model_name:
            return "mistral"
        elif "qwen" in model_name:
            return "qwen"
        elif "deepseek" in model_name:
            return "deepseek"
        else:
            return "generic"