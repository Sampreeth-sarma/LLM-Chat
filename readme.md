# LLM Chat Application Demo (Ollama vs vLLM)

A Streamlit-based chat application to demonstrate and compare **local LLM inference** using:

- **Ollama** – developer-friendly local runtime
- **vLLM** – production-grade inference engine with batching, caching, and concurrency

This demo highlights:
- Chat-based interaction
- Provider and model switching
- Streaming responses
- Performance characteristics (TTFT, tokens/sec, concurrency behavior)

---

## 1. Repository Setup

### Clone the repository
```bash
git clone https://github.com/Sampreeth-sarma/LLM-Chat.git
cd LLM-Chat
```

### Create Virtual Env
```bash
python3 -m venv llm
source llm/bin/activate
```

### Install Python dependencies
```bash
pip install -r requirements.txt
```

## Running the Streamlit App
From project root:
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
```

The app will be available at:
```arduino
http://localhost:8501
```


## Provider Option A: Ollama

### INstall Ollama
macOS: Download from https://ollama.com

Linux: Follow Ollama’s official installation instructions

### Start Ollama server
```bash
ollama serve
```

### Pull a model
```bash
ollama pull llama3.2:1b-instruct
```

### Use Ollama in the app
- Select Model Provider: Ollama
- Choose the model
- Click Initialize Model
- Start chatting

Note: Ollama streams tokens by default
but has limited observability and concurrency support

## Provider Option B: vLLM (Recommended for Performance)
vLLM runs as a separate OpenAI-compatible inference server.
The Streamlit app connects to it over HTTP.

### GPU Notes (Important)
FlashAttention 2 is primarily supported by
NVIDIA's Ampere (RTX 30xx, A100), Ada Lovelace (RTX 40xx), and Hopper (H100) architectures for best performance, requiring CUDA 12.0+, with earlier Turing (RTX 20xx, T4) GPUs also working via FlashAttention 1.x for now. For AMD, it's available on Instinct MI200/MI300 series via ROCm 6.0+

Use TRITON_ATTN or TORCH_SDPA instead if FA2 is not supported

vLLM will automatically fall back to FP16 if BF16 is unsupported

### Install vLLM (Cuda)

__CUDA 13.0__
```bash
pip uninstall vllm -y
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install vllm --extra-index-url https://download.pytorch.org
```
__CUDA 12.1__
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

__VERIFY GPU__
```bash
nvidia-smi
```

### Start vLLM Server

__Example for a small instruct model:__

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --dtype float16 \
  --attention-backend TRITON_ATTN
```

__For newer vllm versions__
```bash
--attention-config.backend FA2
```

### Use vLLM in the app

- Select Model Provider: vLLM
- Select the same model name
- Click Initialize Model
- Start chatting

vLLm has xxcellent TTFT and concurrency, Prefix caching, batching, scheduling, and many more 

## Benchmarking (Optional)

You can benchmark Ollama and vLLM using _bench.py_.

Example runs:
```bash
python bench.py --provider both --concurrency 1 --requests 32 --max_tokens 256
python bench.py --provider both --concurrency 4 --requests 32 --max_tokens 256
python bench.py --provider both --concurrency 16 --requests 32 --max_tokens 256
```

### Key Metrics

- TTFT (Time To First Token)
- Total latency
- Tokens/sec
- Concurrency scaling

### Understanding Results

- Concurrency = 1 → single-user latency
- Higher concurrency → production-like load
- vLLM excels due to:

    - dynamic batching

    - prefix caching

    - KV cache reuse

- Ollama may feel fast in UI but degrades under load

## Common Errors & Fixes
### Prompt too long
```bash
Decoder prompt longer than max_model_len
```
__Fix:__
- Truncate chat history
- Increase --max-model-len (if GPU allows)

### FlashAttention error on T4
```bash
FA2 only supported on compute capability >= 8
```
__Fix:__
- --attention-backend TRITON_ATTN

### GPU memory utilization error
```bash
Free memory < desired GPU utilization
```
__Fix:__
- --gpu-memory-utilization 0.5
- Check running processes:
    - nvidia-smi

### Missing Python headers
```bash
fatal error: Python.h: No such file
```
__Fix (Ubuntu):__
```bash
sudo apt-get install python3-dev build-essential
```