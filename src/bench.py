import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from transformers import AutoTokenizer

_tokenizer_cache = {}

def get_tokenizer(model_name: str):
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
    return _tokenizer_cache[model_name]


@dataclass
class Result:
    provider: str
    model: str
    prompt_id: int
    ok: bool
    error: Optional[str]
    ttft_s: Optional[float]
    total_s: Optional[float]
    completion_tokens: Optional[int]
    tokens_per_s: Optional[float]


def _pctl(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def bench_ollama_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    prompt_id: int,
    temperature: float,
    max_tokens: int,
) -> Result:
    """
    Ollama streaming chat API:
    POST /api/chat  { model, messages, stream:true, options:{...} }
    It streams JSON lines; the final line has done=true and includes eval_count / eval_duration sometimes.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    start = time.perf_counter()
    ttft = None
    completion_tokens = None
    full_text = []

    try:
        async with client.stream("POST", url, json=payload, timeout=None) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                t = time.perf_counter()
                if ttft is None:
                    ttft = t - start

                obj = json.loads(line)

                # Streaming chunks:
                # obj.get("message", {}).get("content", "")
                chunk = (obj.get("message") or {}).get("content") or ""
                if chunk:
                    full_text.append(chunk)

                # Final stats:
                if obj.get("done") is True:
                    # Ollama includes eval_count in some responses
                    completion_tokens = obj.get("eval_count") or obj.get("tokens") or None
                    break

        end = time.perf_counter()
        total = end - start
        tps = (completion_tokens / total) if (completion_tokens and total > 0) else None

        return Result(
            provider="ollama",
            model=model,
            prompt_id=prompt_id,
            ok=True,
            error=None,
            ttft_s=ttft,
            total_s=total,
            completion_tokens=completion_tokens,
            tokens_per_s=tps,
        )
    except Exception as e:
        return Result(
            provider="ollama",
            model=model,
            prompt_id=prompt_id,
            ok=False,
            error=str(e),
            ttft_s=ttft,
            total_s=None,
            completion_tokens=None,
            tokens_per_s=None,
        )


async def bench_vllm_openai_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    prompt_id: int,
    temperature: float,
    max_tokens: int,
) -> Result:
    """
    vLLM OpenAI-compatible streaming benchmark with:
    - TTFT measurement
    - tokenizer-based completion token counting
    - tokens/sec computation
    """

    url = base_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    tokenizer = get_tokenizer(model)

    start = time.perf_counter()
    ttft = None

    generated_chunks: list[str] = []

    try:
        async with client.stream("POST", url, json=payload, timeout=None) as r:
            r.raise_for_status()

            async for raw in r.aiter_lines():
                if not raw:
                    continue

                if not raw.startswith("data: "):
                    continue

                data = raw[len("data: "):].strip()

                if data == "[DONE]":
                    break

                obj = json.loads(data)

                # Extract streamed token text
                delta = (
                    obj.get("choices", [{}])[0]
                       .get("delta", {})
                       .get("content")
                )

                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    generated_chunks.append(delta)

        end = time.perf_counter()
        total_time = end - start

        full_text = "".join(generated_chunks)

        completion_tokens = len(
            tokenizer.encode(full_text, add_special_tokens=False)
        )

        tokens_per_s = (
            completion_tokens / total_time if total_time > 0 else None
        )

        return Result(
            provider="vllm",
            model=model,
            prompt_id=prompt_id,
            ok=True,
            error=None,
            ttft_s=ttft,
            total_s=total_time,
            completion_tokens=completion_tokens,
            tokens_per_s=tokens_per_s,
        )

    except Exception as e:
        return Result(
            provider="vllm",
            model=model,
            prompt_id=prompt_id,
            ok=False,
            error=str(e),
            ttft_s=ttft,
            total_s=None,
            completion_tokens=None,
            tokens_per_s=None,
        )


async def run_benchmark(
    provider: str,
    prompts: List[str],
    concurrency: int,
    ollama_url: str,
    ollama_model: str,
    vllm_url: str,
    vllm_model: str,
    temperature: float,
    max_tokens: int,
) -> List[Result]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Result] = []

    async with httpx.AsyncClient() as client:

        async def one(i: int, prompt: str, prov: str):
            async with sem:
                if prov == "ollama":
                    return await bench_ollama_chat(client, ollama_url, ollama_model, prompt, i, temperature, max_tokens)
                else:
                    return await bench_vllm_openai_chat(client, vllm_url, vllm_model, prompt, i, temperature, max_tokens)

        tasks = []
        if provider in ("ollama", "both"):
            for i, p in enumerate(prompts):
                tasks.append(asyncio.create_task(one(i, p, "ollama")))
        if provider in ("vllm", "both"):
            for i, p in enumerate(prompts):
                tasks.append(asyncio.create_task(one(i, p, "vllm")))

        for t in asyncio.as_completed(tasks):
            results.append(await t)

    return results


def summarize(results: List[Result]) -> None:
    by = {}
    for r in results:
        by.setdefault(r.provider, []).append(r)

    for prov, rs in by.items():
        ok = [x for x in rs if x.ok]
        bad = [x for x in rs if not x.ok]
        ttft = [x.ttft_s for x in ok if x.ttft_s is not None]
        tps = [x.tokens_per_s for x in ok if x.tokens_per_s is not None]
        total = [x.total_s for x in ok if x.total_s is not None]

        print("\n==============================")
        print(f"Provider: {prov}")
        print(f"OK: {len(ok)}  Errors: {len(bad)}")
        if bad:
            print("Sample error:", bad[0].error)

        if ttft:
            print(f"TTFT (s): mean={statistics.mean(ttft):.3f}  p50={_pctl(ttft,0.5):.3f}  p95={_pctl(ttft,0.95):.3f}")
        else:
            print("TTFT (s): n/a")

        if total:
            print(f"Total (s): mean={statistics.mean(total):.3f}  p50={_pctl(total,0.5):.3f}  p95={_pctl(total,0.95):.3f}")
        else:
            print("Total (s): n/a")

        if tps:
            print(f"Tok/s: mean={statistics.mean(tps):.1f}  p50={_pctl(tps,0.5):.1f}  p95={_pctl(tps,0.95):.1f}")
        else:
            print("Tok/s: n/a (enable include_usage, or post-count tokens offline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["ollama", "vllm", "both"], default="both")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=256)

    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--ollama_model", default="llama3.2:1b-instruct-fp16")

    ap.add_argument("--vllm_url", default="http://localhost:8001")
    ap.add_argument("--vllm_model", default="meta-llama/Llama-3.2-1B-Instruct")

    args = ap.parse_args()

    # simple prompt set
    base_prompts = [
        "Summarize the benefits of KV caching in 3 bullet points.",
        "Explain prefix caching to a junior engineer using an analogy.",
        "Write a short Python function that reverses a string and explain it.",
        "Give me a 1-paragraph overview of vLLM vs Ollama.",
    ]
    prompts = [base_prompts[i % len(base_prompts)] + f" (req {i})" for i in range(args.requests)]

    results = asyncio.run(
        run_benchmark(
            provider=args.provider,
            prompts=prompts,
            concurrency=args.concurrency,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            vllm_url=args.vllm_url,
            vllm_model=args.vllm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    summarize(results)


if __name__ == "__main__":
    main()
