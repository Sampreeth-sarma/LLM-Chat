import gc
import inspect
import os
import queue
import threading
import time
import torch
from .base import BaseModelProvider
from huggingface_hub import login
from dotenv import load_dotenv
import logging
import time
import uuid
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

# Set the threshold to DEBUG to see all messages
logging.basicConfig(level=logging.INFO)

load_dotenv()

login(token=os.getenv("hf_token"))


class VLLMProvider(BaseModelProvider):
    def __init__(self, model_name, gpu_id=0, async_flow=True, tensor_parallel_size=1,
                 attention_method="auto", **kwargs):
        super().__init__()
        
        logging.info("VLLM Log: Beginning Model Initialization")
        # ---- GPU selection (must be set before CUDA context is created) ----
        if tensor_parallel_size == 1 and gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # ---- Detect GPU capability ----
        major, minor = (0, 0)
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
        except Exception:
            pass

        requested = (attention_method or "auto").lower()

        # On Tesla T4 (SM75 / major < 8), FA2 won't work.
        # Prefer TORCH_SDPA (most compatible) or TRITON_ATTN.
        if major < 8:
            os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        else:
            # Ampere+ can often use FLASH_ATTN
            if requested in ("flash_attention_2", "fa2", "flashattn2"):
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
            elif requested in ("triton", "triton_attn"):
                os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
            elif requested in ("sdpa", "torch_sdpa"):
                os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
            else:
                # keep auto
                pass
        
        logging.info(f"\n\nVLLM Log: ATTENTION BACKEND SELECTED: {os.getenv('VLLM_ATTENTION_BACKEND')}\n\n")

        # ---- Import vLLM only after env vars are set ----
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
        self.max_gpu_utilization = kwargs.get("max_gpu_utilization")
        self.max_model_len = kwargs.get("max_model_len")

        # ---- IMPORTANT: disable torch.compile / cudagraph for compatibility ----
        # vLLM docs: enforce_eager=True disables vLLM-torch.compile integration. :contentReference[oaicite:8]{index=8}
        if not async_flow:
            enforce_eager = kwargs.get("enforce_eager", True)

            llm_kwargs = {
                "model": model_name,
                "tensor_parallel_size": tensor_parallel_size,
                "enforce_eager": enforce_eager,
                # Prefix caching (what you wanted to toggle conceptually)
            }

            # Optional knobs you might already pass
            if kwargs.get("dtype"):
                llm_kwargs["dtype"] = kwargs["dtype"]
            if kwargs.get("max_model_len"):
                llm_kwargs["max_model_len"] = kwargs["max_model_len"]
            if kwargs.get("gpu_memory_utilization"):
                llm_kwargs["gpu_memory_utilization"] = kwargs["gpu_memory_utilization"]
            if kwargs.get("kv_cache_dtype"):
                llm_kwargs["kv_cache_dtype"] = kwargs["kv_cache_dtype"]

            try:
                # Filter only supported kwargs based on the actual installed vLLM signature
                init_sig = inspect.signature(self.LLM.__init__)
                allowed = set(init_sig.parameters.keys())
                filtered = {k: v for k, v in llm_kwargs.items() if k in allowed}

                logging.info("VLLM Log: Model Init")
                self.llm = self.LLM(**filtered)
            except Exception as e:
                print(f"Error initializing vLLM: {e}")
                raise
        else:
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                enforce_eager=False,  # stable startup; also used in official example :contentReference[oaicite:3]{index=3}
            )
            # If you already have tensor_parallel_size stored, you can set it:
            if hasattr(self, "tensor_parallel_size") and self.tensor_parallel_size:
                engine_args.tensor_parallel_size = self.tensor_parallel_size

            if hasattr(self, "max_model_len") and self.max_model_len:
                engine_args.max_model_len = self.max_model_len
            if hasattr(self, "dtype") and self.dtype:
                engine_args.dtype = self.dtype
            if hasattr(self, "max_gpu_utilization") and self.max_gpu_utilization:
                engine_args.gpu_memory_utilization = self.max_gpu_utilization

            logging.info(f"ENGINE ARGS: \n\n{engine_args}\n\n")
            self.async_llm = AsyncLLM.from_engine_args(engine_args)
            self._ensure_async_loop()

    def stream_sync(self, **gen_kwargs):
        """
        Sync generator that yields streaming chunks from generateAsync()
        using the provider's persistent asyncio loop.
        """
        self._ensure_async_loop()

        q = queue.Queue()
        SENTINEL = object()

        async def _runner():
            try:
                async for chunk in self.generateAsync(**gen_kwargs):
                    q.put(chunk)
            except Exception as e:
                q.put(e)
            finally:
                q.put(SENTINEL)

        # Schedule the async runner on the persistent loop
        asyncio.run_coroutine_threadsafe(_runner(), self._loop)

        # Yield chunks as they arrive
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _ensure_async_loop(self):
        if getattr(self, "_loop", None) and getattr(self, "_loop_thread", None):
            return

        self._loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, daemon=True)
        self._loop_thread.start()

    
    def generate(self, messages, temperature=0.7, max_tokens=2048, top_p=1.0, top_k=40, stop=None, **kwargs):
        """Generate response using vLLM with streaming"""
        logging.info("VLLM Log: Starting Generation")
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
        outputs = self.llm.generate(prompt, sampling_params,
                                    # stream=True
                                )
        
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
            
            token_count = len(current_output.token_ids) if isinstance(current_output.token_ids, list) else current_output.token_ids.shape[0]
            yield generated_text
        
        # Track metrics after generation completes
        self._track_generation_metrics(start_time, token_count)
    
    async def generateAsync(
        self,
        messages,
        temperature=0.7,
        max_tokens=2048,
        top_p=1.0,
        top_k=40,
        stop=None,
        **kwargs
    ):
        """
        vLLM v0.13 async streaming (V1 engine) using AsyncLLM + DELTA output mode.
        Yields incremental text deltas as they are generated.
        """

        # vLLM v0.13 official example uses these imports. :contentReference[oaicite:1]{index=1}

        prompt = self._format_messages(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop if stop else None,
            output_kind=RequestOutputKind.DELTA,  # <- key for true streaming deltas :contentReference[oaicite:2]{index=2}
        )

        # Create/reuse AsyncLLM engine to avoid reloading weights every call.
        if not hasattr(self, "async_llm") or self.async_llm is None:
            # Keep args minimal and only use knobs you already have stored on self.
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                enforce_eager=False,  # stable startup; also used in official example :contentReference[oaicite:3]{index=3}
            )
            # If you already have tensor_parallel_size stored, you can set it:
            if hasattr(self, "tensor_parallel_size") and self.tensor_parallel_size:
                engine_args.tensor_parallel_size = self.tensor_parallel_size

            # If you already manage max_model_len / dtype in __init__, and have them stored,
            # you can set them here too, but only if you are sure these fields exist:
            if hasattr(self, "max_model_len") and self.max_model_len:
                engine_args.max_model_len = self.max_model_len
            if hasattr(self, "dtype") and self.dtype:
                engine_args.dtype = self.dtype
            if hasattr(self, "max_gpu_utilization") and self.max_gpu_utilization:
                engine_args.gpu_memory_utilization = self.max_gpu_utilization
            logging.info(f"ENGINE ARGS: {engine_args}")
            self.async_llm = AsyncLLM.from_engine_args(engine_args)

        start_time = time.time()
        first_token = True
        token_count = 0

        request_id = f"stream-{uuid.uuid4().hex}"

        try:
            async for output in self.async_llm.generate(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            ):
                # In DELTA mode, completion.text is only the new chunk since last yield. :contentReference[oaicite:4]{index=4}
                for completion in output.outputs:
                    delta = completion.text or ""
                    if delta:
                        if first_token:
                            first_token = False
                            self.metrics["latency"].append(time.time() - start_time)
                            if len(self.metrics["latency"]) > 10:
                                self.metrics["latency"] = self.metrics["latency"][-10:]
                        yield delta

                    # best-effort token count if available
                    try:
                        if getattr(completion, "token_ids", None) is not None:
                            token_count = len(completion.token_ids)
                    except Exception:
                        pass

                if getattr(output, "finished", False):
                    break

            # Final metrics (same behavior as your sync generate)
            self._track_generation_metrics(start_time, token_count)

        except asyncio.CancelledError:
            # If Streamlit cancels mid-stream, just propagate.
            raise
        
    @classmethod
    def list_models(cls):
        """Return common models supported by vLLM"""
        # This is a static list as vLLM doesn't maintain an API to list models
        return [
            "meta-llama/Llama-3.2-1b-Instruct",
            "meta-llama/Llama-3.2-3b-instruct",
            "meta-llama/Llama-3.2-1b",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen-7B-Chat",
            "Qwen/Qwen-14B-Chat",
            "deepseek-ai/deepseek-coder-7b-instruct",
            "deepseek-ai/deepseek-llm-7b-chat"
        ]
    
    @classmethod
    def list_attention_methods(cls):
        """List available attention methods in vLLM"""
        return [
            "auto",
            "xformers",
            "flash_attention",
            "flash_attention_2",
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
    
    def set_attention_method(self, method):
        """Change attention method - requires reinitializing the model"""
        if method == self.attention_method:
            return False
        
        self.attention_method = method
        # Need to reinitialize the model with the new attention method
        # This will be handled in the app by reinitializing the provider
        return True
    
    def close(self):
        # Try to shut down async engine cleanly
        try:
            if getattr(self, "async_llm", None) is not None:
                # vLLM internals can vary; try common shutdown hooks
                engine = getattr(self.async_llm, "engine", None)
                if engine is not None:
                    # Best-effort: stop background workers/executor
                    executor = getattr(engine, "executor", None)
                    if executor is not None:
                        shutdown = getattr(executor, "shutdown", None)
                        if callable(shutdown):
                            shutdown()

                self.async_llm = None
        except Exception:
            pass

        # Drop any sync engine too (if you ever kept one)
        try:
            self.llm = None
        except Exception:
            pass

        # Force python to release objects and ask CUDA to release cached blocks
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    