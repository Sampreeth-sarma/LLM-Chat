import time
import torch
from .base import BaseModelProvider

class VLLMProvider(BaseModelProvider):
    def __init__(self, model_name, gpu_id=0, tensor_parallel_size=1, 
                 attention_method="auto", **kwargs):
        """Initialize vLLM with specified model and GPU configuration"""
        super().__init__()
        
        try:
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError("vLLM is not installed. Install it with 'pip install vllm'.")
        
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.tensor_parallel_size = tensor_parallel_size
        self.attention_method = attention_method
        self.model_family = self._detect_model_family(model_name)
        
        # Initialize vLLM with the appropriate configuration
        try:
            kwargs_config = {
                "tensor_parallel_size": tensor_parallel_size,
            }
            
            # Add GPU selection if specified
            if tensor_parallel_size == 1 and gpu_id is not None:
                kwargs_config["gpu_ids"] = [gpu_id]
                
            # Add attention method configuration
            if attention_method != "auto":
                kwargs_config["attention_mechanism"] = attention_method
                
            # Add KV cache configuration if specified
            if kwargs.get("kv_cache_dtype"):
                kwargs_config["kv_cache_dtype"] = kwargs.get("kv_cache_dtype")
                
            self.llm = self.LLM(
                model=model_name,
                disable_kv_cache_reuse=not self.use_kv_cache,  # KV-cache toggle
                **kwargs_config,
            )
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            raise
    
    def generate(self, messages, temperature=0.7, max_tokens=2048, top_p=1.0, 
                top_k=40, stop=None, **kwargs):
        """Generate response using vLLM with streaming"""
        prompt = self._format_messages(messages)
        
        # Set sampling parameters
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop if stop else None
        )
        
        # Measure start time for performance tracking
        start_time = time.time()
        first_token = True
        token_count = 0
        
        # Generate and stream the response
        outputs = self.llm.generate(prompt, sampling_params, stream=True)
        
        for output in outputs:
            current_output = output.outputs[0]
            generated_text = current_output.text
            
            # Track first token latency
            if first_token and generated_text.strip():
                first_token = False
                self.metrics["latency"].append(time.time() - start_time)
                # Keep only last 10 latencies
                if len(self.metrics["latency"]) > 10:
                    self.metrics["latency"] = self.metrics["latency"][-10:]
            
            token_count = current_output.token_ids.shape[0]
            yield generated_text
        
        # Track metrics after generation completes
        self._track_generation_metrics(start_time, token_count)
    
    def list_models(self):
        """Return common models supported by vLLM"""
        # This is a static list as vLLM doesn't maintain an API to list models
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-3-8b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen-7B-Chat",
            "Qwen/Qwen-14B-Chat",
            "deepseek-ai/deepseek-coder-7b-instruct",
            "deepseek-ai/deepseek-llm-7b-chat"
        ]
    
    def list_attention_methods(self):
        """List available attention methods in vLLM"""
        return [
            "auto", 
            "flash_attention", 
            "flash_attention_2",
            "paged_attention", 
            "eager"
        ]
        
    def _format_messages(self, messages):
        """Format messages based on model family"""
        model_family = self._detect_model_family(self.model_name)
        
        if "llama" in model_family:
            return self._format_llama_messages(messages)
        elif "mistral" in model_family:
            return self._format_mistral_messages(messages)
        elif "qwen" in model_family:
            return self._format_qwen_messages(messages)
        elif "deepseek" in model_family:
            return self._format_deepseek_messages(messages)
        else:
            # Default formatting
            return self._format_generic_messages(messages)
    
    # Implement the same formatting methods as in OllamaProvider
    # _format_llama_messages, _format_mistral_messages, etc.
    # (These are identical to the ones in OllamaProvider so I'm not duplicating them here)
            
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
    
    def set_attention_method(self, method):
        """Change attention method - requires reinitializing the model"""
        if method == self.attention_method:
            return False
        
        self.attention_method = method
        # Need to reinitialize the model with the new attention method
        # This will be handled in the app by reinitializing the provider
        return True