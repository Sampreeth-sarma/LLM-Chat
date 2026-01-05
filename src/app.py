import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st
import torch

from model_providers.ollama import OllamaProvider
from model_providers.vllm import VLLMProvider
from utils.device_utils import get_recommended_config, get_system_info
from utils.prompt_manager import PromptManager
from utils.visualization import PerformanceVisualizer

logging.basicConfig(level=logging.INFO)


# -----------------------------
# App setup helpers
# -----------------------------

def load_css(path: str = "style.css") -> None:
    """Load CSS from file if present."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        # CSS is optional; keep UI functional even if missing.
        pass


def configure_page() -> None:
    st.set_page_config(page_title="LLM Demo", page_icon="🤖", layout="wide")


def init_session_state() -> None:
    """Initialize all session state variables used in the app."""
    defaults = {
        "pending_user_input": None,
        "messages": [],
        "chat_history": [],
        "is_streaming": False,
        "streaming_text": "",
        "draft_next": "",
        "clear_draft_next": False,
        "queued_next": None,
        "providers": {},
        "active_provider": None,
        "prompt_manager": PromptManager(),
        "performance_visualizer": PerformanceVisualizer(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def has_active_model() -> bool:
    key = st.session_state.get("active_provider")
    if not key:
        return False
    return key in st.session_state.get("providers", {})


def cleanup_active_provider() -> None:
    logging.info("CLEANUP LOG: Cleanup in progress")
    key = st.session_state.get("active_provider")
    if not key:
        return
    logging.info(f"CLEANUP LOG: Cleanup in progress for {key}")
    old = st.session_state.providers.get(key)
    if old:
        try:
            old.close()
        except Exception:
            pass
    st.session_state.providers.pop(key, None)
    st.session_state.active_provider = None


# -----------------------------
# vLLM async->sync wrapper (kept for compatibility)
# NOTE: In your newer setup you may prefer provider.stream_sync().
# -----------------------------
def stream_vllm_async(provider, **gen_kwargs):
    """
    Convert provider.generateAsync(...) (async generator) into a sync generator.
    This avoids asyncio.run() inside Streamlit and keeps app code simple.
    """
    async def agen():
        async for chunk in provider.generateAsync(**gen_kwargs):
            yield chunk

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        aiter = agen()

        while True:
            try:
                chunk = loop.run_until_complete(aiter.__anext__())
                yield chunk
            except StopAsyncIteration:
                break
    finally:
        try:
            loop.close()
        except Exception:
            pass


# -----------------------------
# UI pieces
# -----------------------------
def render_header() -> None:
    st.title("🤖 LLM Chat Application Demo")


def detect_model_family(selected_model: str) -> str:
    sm = (selected_model or "").lower()
    if "llama" in sm:
        return "llama"
    if "mistral" in sm:
        return "mistral"
    if "qwen" in sm:
        return "qwen"
    if "deepseek" in sm:
        return "deepseek"
    return "generic"


def get_available_models(provider_option: str) -> Tuple[list, str]:
    """Returns (available_models, selected_model)."""
    if provider_option == "Ollama":
        try:
            ollama_provider = OllamaProvider()
            available_models = ollama_provider.list_models()
        except Exception:
            available_models = ["N/A"]
        selected_model = st.selectbox("Model", available_models)
        return available_models, selected_model

    # vLLM
    try:
        available_models = VLLMProvider.list_models()
    except Exception:
        available_models = ["N/A"]
    selected_model = st.selectbox("Model", available_models)
    return available_models, selected_model


def render_generation_params(provider_option: str) -> Dict[str, Any]:
    """Render generation parameters UI and return values dict."""
    params: Dict[str, Any] = {}

    with st.expander("Generation Parameters"):
        params["temperature"] = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

        col1_inner, col2_inner = st.columns(2)
        with col1_inner:
            params["top_p"] = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        with col2_inner:
            params["top_k"] = st.slider("Top-k", min_value=1, max_value=100, value=40, step=1)

        params["max_tokens"] = st.number_input("Max Tokens", min_value=10, max_value=4096, value=2048, step=10)

        # Provider-specific
        if provider_option == "vLLM":
            try:
                attention_methods = VLLMProvider.list_attention_methods()
            except Exception:
                attention_methods = ["N/A"]

            params["attention_method"] = st.selectbox(
                "Attention Method",
                attention_methods,
                help="Different attention implementations (Flash Attention is fastest on modern GPUs)"
            )

            if torch.cuda.is_available():
                available_gpus = list(range(torch.cuda.device_count()))
                params["selected_gpu"] = st.selectbox(
                    "GPU",
                    available_gpus,
                    format_func=lambda x: f"GPU {x}: {torch.cuda.get_device_name(x)}"
                )
                params["tensor_parallel"] = st.slider(
                    "Tensor Parallel Size",
                    max_value=len(available_gpus),
                    value=1
                )
            else:
                st.warning("No CUDA-capable GPUs detected")
                params["selected_gpu"] = None
                params["tensor_parallel"] = 1

    return params


def render_prompt_section() -> Tuple[str, str]:
    """Render system prompt UI and return (selected_template, system_prompt)."""
    st.subheader("System Prompt")
    template_options = st.session_state.prompt_manager.list_templates()
    selected_template = st.selectbox("Select Template", template_options)

    system_prompt = st.text_area(
        "Customize System Prompt",
        value=st.session_state.prompt_manager.get_template(selected_template),
        height=100
    )
    return selected_template, system_prompt


def initialize_provider(provider_option: str, selected_model: str, kv_cache_enabled: bool, gen_params: Dict[str, Any]) -> None:
    """Initialize a provider and store it in session_state."""
    cleanup_active_provider()
    logging.info(f"INIT LOG: Initializing {selected_model}")

    with st.spinner(f"Initializing {selected_model}..."):
        try:
            provider_key = f"{provider_option}_{selected_model}"

            if provider_option == "Ollama":
                logging.info("INIT LOG: Ollama Provider")
                provider = OllamaProvider(model_name=selected_model)

            elif provider_option == "vLLM":
                logging.info("INIT LOG: VLLM Provider")
                selected_gpu = gen_params.get("selected_gpu") if torch.cuda.is_available() else None
                tensor_parallel = gen_params.get("tensor_parallel", 1) if torch.cuda.is_available() else 1
                attention_method = gen_params.get("attention_method", "N/A")

                # NOTE: keeping your existing param name to avoid behavior changes.
                provider = VLLMProvider(
                    model_name=selected_model,
                    gpu_id=selected_gpu,
                    tensor_parallel_size=tensor_parallel,
                    attention_method=attention_method,
                    max_gpu_utilization=0.5,
                    # kv_cache_dtype="fp8",
                    max_model_len=4096,
                )
            else:
                raise ValueError(f"Unknown provider option: {provider_option}")

            provider.set_kv_cache(kv_cache_enabled)

            st.session_state.providers[provider_key] = provider
            st.session_state.active_provider = provider_key
            st.success(f"Model {selected_model} initialized successfully!")

        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")


def unload_active_provider() -> None:
    """Unload the active model/provider from session state (best-effort close)."""
    active_key = st.session_state.get("active_provider")
    provider = st.session_state.providers.get(active_key) if active_key else None

    try:
        if provider and hasattr(provider, "close"):
            provider.close()
    except Exception:
        pass

    if active_key:
        st.session_state.providers.pop(active_key, None)
    st.session_state.active_provider = None

    logging.info("UNLOAD LOG: Model unloaded from this session.")
    st.success("Model unloaded from this session.")
    st.rerun()


def get_active_provider():
    key = st.session_state.get("active_provider")
    if not key:
        return None
    return st.session_state.providers.get(key)


def render_settings_panel() -> Dict[str, Any]:
    """Render right-side settings panel and return config dict used in chat generation."""
    config: Dict[str, Any] = {}

    st.markdown('<div class="sticky-right-panel">', unsafe_allow_html=True)
    st.header("Model Settings")

    config["provider_option"] = st.selectbox(
        "Model Provider",
        ["Ollama", "vLLM"],
        help="Select the model provider"
    )

    _, selected_model = get_available_models(config["provider_option"])
    config["selected_model"] = selected_model

    config["kv_cache_enabled"] = st.toggle(
        "Enable Prefix Caching",
        value=True,
        help="Prefix caching reuses the KV cache for shared prompt prefixes across requests (vLLM). Helps TTFT + throughput when prompts share a prefix.)"
    )

    config["model_family"] = detect_model_family(selected_model)
    st.caption(f"Detected model family: {config['model_family']}")

    _, config["system_prompt"] = render_prompt_section()

    gen_params = render_generation_params(config["provider_option"])
    config.update(gen_params)

    if st.button("Initialize Model"):
        initialize_provider(config["provider_option"], selected_model, config["kv_cache_enabled"], config)

    unload_disabled = not has_active_model()
    if st.button("Unload Model", disabled=unload_disabled):
        unload_active_provider()

    # Keep provider KV cache toggle in sync (if provider exists)
    if has_active_model():
        active_provider = get_active_provider()
        if active_provider and getattr(active_provider, "use_kv_cache", None) != config["kv_cache_enabled"]:
            active_provider.set_kv_cache(config["kv_cache_enabled"])
            st.info(f"Prefix Caching {'enabled' if config['kv_cache_enabled'] else 'disabled'}")

    st.markdown('</div>', unsafe_allow_html=True)
    return config


# -----------------------------
# Chat rendering + streaming
# -----------------------------
def render_chat_history() -> None:
    # Display existing messages (skip system message)
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def maybe_promote_pending_user_input() -> None:
    """If there is pending user input and we're not streaming, push it into messages and start streaming."""
    if st.session_state.pending_user_input and not st.session_state.is_streaming:
        st.session_state.messages.append({"role": "user", "content": st.session_state.pending_user_input})
        st.session_state.pending_user_input = None
        st.session_state.is_streaming = True
        st.session_state.streaming_text = ""
        st.rerun()


def stream_assistant_response(config: Dict[str, Any]) -> None:
    """Stream assistant response while keeping an input box visible."""
    with st.chat_message("assistant"):
        if not has_active_model():
            st.error("Please initialize a model first using the settings panel.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Please initialize a model first using the settings panel."}
            )
            st.session_state.is_streaming = False
            st.session_state.streaming_text = ""
            st.rerun()
            return

        # Show an input box while generating (draft next)
        if st.session_state.clear_draft_next:
            st.session_state.draft_next = ""
            st.session_state.clear_draft_next = False

        st.text_input(
            "Type your message here...",
            key="draft_next",
            label_visibility="collapsed",
            placeholder="Assistant is responding… type your next message",
        )

        message_placeholder = st.empty()
        full_response = ""

        provider = get_active_provider()
        st.session_state.performance_visualizer.start_measurement()

        try:
            # Choose streaming iterator
            if config.get("provider_option") == "Ollama":
                stream_iter = provider.generate(
                    messages=st.session_state.messages,
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 2048),
                    top_p=config.get("top_p", 0.9),
                    top_k=config.get("top_k", 40),
                )
            else:
                # Preferred: provider.stream_sync() if implemented; else fallback wrapper
                if hasattr(provider, "stream_sync"):
                    stream_iter = provider.stream_sync(
                        messages=st.session_state.messages,
                        temperature=config.get("temperature", 0.7),
                        max_tokens=config.get("max_tokens", 2048),
                        top_p=config.get("top_p", 0.9),
                        top_k=config.get("top_k", 40),
                    )
                else:
                    stream_iter = stream_vllm_async(
                        provider,
                        messages=st.session_state.messages,
                        temperature=config.get("temperature", 0.7),
                        max_tokens=config.get("max_tokens", 2048),
                        top_p=config.get("top_p", 0.9),
                        top_k=config.get("top_k", 40),
                    )

            for response_chunk in stream_iter:
                full_response += response_chunk
                st.session_state.streaming_text = full_response
                message_placeholder.markdown(full_response + "▌")
                st.session_state.performance_visualizer.add_token()

            st.session_state.performance_visualizer.end_measurement(provider.use_kv_cache)
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            full_response = f"Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Finish streaming
        st.session_state.is_streaming = False
        st.session_state.streaming_text = ""

        # If user typed something during streaming, queue it
        if st.session_state.get("draft_next"):
            st.session_state.queued_next = st.session_state.draft_next
            st.session_state.clear_draft_next = True

        st.rerun()


def render_chat_input() -> None:
    """Bottom chat input (only when not streaming)."""
    if st.session_state.is_streaming:
        return

    if st.session_state.queued_next:
        user_input = st.session_state.queued_next
        st.session_state.queued_next = None
    else:
        user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.pending_user_input = user_input
        st.rerun()


def render_auto_scroll_js() -> None:
    st.markdown(
        """
        <script>
        const chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """,
        unsafe_allow_html=True
    )


def render_clear_chat_button(system_prompt: str) -> None:
    if len(st.session_state.messages) > 1 and st.button("Clear Chat History"):
        system_message = st.session_state.messages[0] if st.session_state.messages else {"role": "system", "content": system_prompt}
        st.session_state.messages = [system_message]
        st.rerun()


def render_chat_tab() -> None:
    col_left, col_right = st.columns([3, 1])

    with col_right:
        config = render_settings_panel()

    with col_left:
        # Ensure system message exists (keeps behavior consistent)
        if not st.session_state.messages:
            st.session_state.messages = [{"role": "system", "content": config.get("system_prompt", "")}]

        # Show chat history
        with st.container():
            st.write("")
            render_chat_history()
            st.write("")
            st.write("")

        st.write("")

        # Promote any queued pending user msg into messages before streaming
        maybe_promote_pending_user_input()

        # Stream assistant response (above bottom input)
        if st.session_state.is_streaming:
            stream_assistant_response(config)

        # Bottom input
        render_chat_input()

        render_auto_scroll_js()

        render_clear_chat_button(config.get("system_prompt", ""))


def render_performance_tab() -> None:
    st.header("Performance Comparison")

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.subheader("Performance Settings")

        if st.button("Run Performance Test", help="Runs a standardized test to compare performance with and without KV cache"):
            st.info("Running performance test...")

        if st.button("Clear Performance Data"):
            st.session_state.performance_visualizer.clear_data()
            st.success("Performance data cleared")

    with col_main:
        try:
            st.session_state.performance_visualizer.visualize_performance(st.container())
        except Exception:
            st.info("No performance data available yet. Generate text first or run a performance test.")

    st.subheader("What is KV Caching?")
    st.markdown(
        """
        **Key-Value Caching** is an optimization technique used in transformer-based language models to avoid redundant computation:

        - **Without Prefix Caching**: Prefix reuse is disabled; repeated prompts will recompute the shared prompt prefix.
        - **With Prefix Caching**: vLLM can reuse cached KV blocks for the shared prefix, reducing TTFT and improving throughput.

        **Benefits:**
        1. Significantly faster generation (typically 2-4x speedup)
        2. Reduced computational overhead
        3. Lower memory bandwidth usage

        **Trade-offs:**
        1. Increased memory usage (storing the KV cache)
        2. Can limit context length for very long sequences
        """
    )

    st.subheader("Different Attention Implementation Methods")
    st.markdown(
        """
        Modern LLMs use various attention implementation strategies:

        - **Standard Attention**: The original implementation - accurate but slower
        - **Flash Attention**: Optimized implementation that maximizes GPU utilization
        - **Flash Attention 2**: Improved version with better performance
        - **Group Query Attention (GQA)**: Reduces computation by sharing keys and values across multiple query heads
        - **Multi-Query Attention (MQA)**: Similar to GQA but more aggressive sharing
        - **Sliding Window Attention**: Restricts attention to a local window, reducing complexity

        These optimizations enable running larger models more efficiently.
        """
    )


def render_hardware_tab() -> None:
    st.header("System Hardware Information")

    try:
        system_info = get_system_info()
    except Exception:
        system_info = {
            "platform": "Information not available",
            "python_version": "Information not available",
            "cpu": {"processor": "Information not available", "cores": 0, "threads": 0},
            "memory": {"available": 0, "total": 0},
            "gpu": None,
        }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CPU &amp; Memory")
        st.write(f"**Platform:** {system_info['platform']}")
        st.write(f"**Python Version:** {system_info['python_version']}")
        st.write(f"**CPU:** {system_info['cpu']['processor']}")
        st.write(f"**Cores/Threads:** {system_info['cpu']['cores']}/{system_info['cpu']['threads']}")
        st.write(f"**Memory:** {system_info['memory']['available']}GB available / {system_info['memory']['total']}GB total")

    with col2:
        st.subheader("GPU Information")
        if system_info.get("gpu"):
            for i, gpu in enumerate(system_info["gpu"]):
                st.write(f"**GPU {i}:** {gpu.get('name', 'Unknown')}")
                if "memory_free" in gpu:
                    st.write(f"   - Memory: {gpu['memory_free']}GB free / {gpu['memory_total']}GB total")
                if "load" in gpu:
                    st.write(f"   - Load: {gpu['load']}%")
                    st.progress(gpu["load"] / 100)
        else:
            st.write("**GPU:** None detected")

    st.subheader("Recommended Configurations")
    try:
        recommended_config = get_recommended_config()
        st.json(recommended_config)
    except Exception:
        st.error("Could not generate recommended configurations")

    st.markdown(
        """
        ### Model Size Guidelines

        | Model Type | Parameters | Min VRAM | Recommended VRAM |
        |------------|------------|----------|------------------|
        | Small      | 7B-8B      | 8GB      | 12GB             |
        | Medium     | 13B-14B    | 16GB     | 24GB             |
        | Large      | 30B-70B    | 32GB     | 48GB+            |
        """
    )


def render_footer() -> None:
    st.markdown("---")
    st.caption("Built with Streamlit. Created for demonstration purposes.")

    if has_active_model():
        try:
            provider = get_active_provider()
            metrics = provider.get_performance_metrics()
            st.caption(f"Average generation speed: {metrics['avg_speed']:.2f} tokens/sec")
        except Exception:
            st.caption("Performance metrics not available")

    st.markdown(
        """
        <div style="position: absolute; top: 10px; right: 15px; z-index: 1000;">
            <a style="text-decoration: none; color: #0366d6; font-weight: 600;" href="https://github.com/sampreeth-sarma/LLM-Chat">
                📂 View on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    load_css()
    configure_page()
    init_session_state()
    render_header()

    tab1, tab2, tab3 = st.tabs(["Chat", "Performance Comparison", "Hardware Info"])

    with tab1:
        render_chat_tab()

    with tab2:
        render_performance_tab()

    with tab3:
        render_hardware_tab()

    render_footer()


if __name__ == "__main__":
    main()
