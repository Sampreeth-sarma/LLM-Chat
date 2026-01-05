import streamlit as st
import torch
import time
import pandas as pd

from model_providers.ollama import OllamaProvider
from model_providers.vllm import VLLMProvider
from utils.prompt_manager import PromptManager
from utils.device_utils import get_system_info, get_recommended_config
from utils.visualization import PerformanceVisualizer

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="LLM Chat Demo", page_icon="🤖", layout="wide")

# ----------------------------
# Clean CSS (NO HTML-escaped tags)
# ----------------------------
st.markdown(
    """
    <style>
    /* Give extra breathing room at bottom so input doesn't overlap */
    div.block-container { padding-bottom: 120px; }

    /* OPTIONAL: If you previously used fixed chat input, REMOVE it.
       Streamlit's native chat_input already sits at the bottom of the page. */
    /* .stChatInputContainer { position: fixed; bottom: 20px; width: calc(100% - 420px); } */

    section[data-testid="stChatMessageContainer"] { margin-bottom: 14px; }

    /* Slightly nicer sidebar spacing */
    [data-testid="stSidebar"] { padding-top: 18px; }

    /* Make chat area feel like a panel */
    .chat-panel {
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Session state init
# ----------------------------
if "prompt_manager" not in st.session_state:
    st.session_state.prompt_manager = PromptManager()

if "performance_visualizer" not in st.session_state:
    st.session_state.performance_visualizer = PerformanceVisualizer()

if "providers" not in st.session_state:
    st.session_state.providers = {}

if "active_provider" not in st.session_state:
    st.session_state.active_provider = None

if "messages" not in st.session_state:
    # Create a default system prompt on first load
    default_system = "You are a helpful AI assistant."
    st.session_state.messages = [{"role": "system", "content": default_system}]

# Two-phase streaming state (THIS is the key fix)
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# ----------------------------
# Sidebar: Model settings
# ----------------------------
with st.sidebar:
    st.title("Model Settings")

    # Provider selection
    provider_option = st.selectbox(
        "Model Provider",
        ["Ollama", "vLLM"],
        help="Select the model provider backend",
    )

    # Model list selection
    if provider_option == "Ollama":
        try:
            tmp_provider = OllamaProvider()
            available_models = tmp_provider.list_models()
        except Exception:
            available_models = ["llama2", "llama3", "mistral", "qwen", "deepseek-coder"]

        selected_model = st.selectbox("Model", available_models)

    else:
        # vLLM: these are typical HF repo ids (your VLLMProvider likely expects these)
        try:
            tmp_provider = VLLMProvider("meta-llama/Llama-2-7b-chat-hf")
            available_models = tmp_provider.list_models()
        except Exception:
            available_models = [
                "meta-llama/Llama-2-7b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "Qwen/Qwen-7B-Chat",
            ]

        selected_model = st.selectbox("Model", available_models)

    # KV cache toggle (works best on vLLM, but you can keep it for both)
    use_kv_cache = st.toggle("Enable KV Cache", value=True)

    st.divider()

    # System prompt / templates
    st.subheader("System Prompt")

    template_names = st.session_state.prompt_manager.list_templates()
    selected_template = st.selectbox("Select Template", template_names)

    system_prompt = st.text_area(
        "Customize System Prompt",
        value=st.session_state.prompt_manager.get_template(selected_template),
        height=120,
    )

    st.divider()

    # Generation params
    with st.expander("Generation Parameters", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
            top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.01)
        with col_b:
            top_k = st.slider("Top-k", 1, 100, 40, 1)

        max_tokens = st.number_input("Max Tokens", min_value=10, max_value=4096, value=2048, step=10)

        # vLLM extras (optional)
        attention_method = "auto"
        selected_gpu = 0
        if provider_option == "vLLM":
            try:
                tmp_provider = VLLMProvider("meta-llama/Llama-2-7b-chat-hf")
                attention_methods = tmp_provider.list_attention_methods()
            except Exception:
                attention_methods = ["auto", "flash_attention", "flash_attention_2", "eager"]

            attention_method = st.selectbox(
                "Attention Method",
                attention_methods,
                help="Flash Attention is fastest on modern GPUs",
            )

            if torch.cuda.is_available():
                gpus = list(range(torch.cuda.device_count()))
                selected_gpu = st.selectbox(
                    "GPU",
                    gpus,
                    format_func=lambda idx: f"GPU {idx}: {torch.cuda.get_device_name(idx)}",
                )
            else:
                st.info("CUDA not available. vLLM may not work without a GPU.")

    st.divider()

    # Initialize model button
    if st.button("Initialize Model"):
        try:
            if provider_option == "Ollama":
                provider = OllamaProvider(model_name=selected_model)
            else:
                provider = VLLMProvider(
                    model=selected_model,
                    attention_method=attention_method,
                    gpu_id=selected_gpu,
                )

            st.session_state.providers[provider_option] = provider
            st.session_state.active_provider = provider_option

            # Update system message in chat history to match UI
            if len(st.session_state.messages) == 0 or st.session_state.messages[0]["role"] != "system":
                st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                st.session_state.messages[0]["content"] = system_prompt

            st.success(f"Initialized {provider_option} with model: {selected_model}")
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")

# ----------------------------
# Main area: Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Chat", "Performance Comparison", "Hardware Info"])

# ----------------------------
# Tab 1: Chat
# ----------------------------
with tab1:
    col_left, col_right = st.columns([3.2, 1.4], gap="large")

    # Right panel: show current config / helper
    with col_right:
        st.header("Model Settings")
        st.caption(f"Provider: **{st.session_state.active_provider or 'Not initialized'}**")
        st.caption(f"KV Cache: **{'On' if use_kv_cache else 'Off'}**")

        st.subheader("System Prompt")
        st.caption("This is sent as the first message in every conversation.")
        st.code(system_prompt)

        st.divider()
        if st.button("Clear Chat History"):
            # Keep system prompt
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            st.session_state.pending_user_input = None
            st.session_state.is_streaming = False
            st.rerun()

    # Left panel: chat UI
    with col_left:
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        st.title("🤖 LLM Chat Application Demo")

        # Ensure system prompt always matches sidebar
        if len(st.session_state.messages) == 0 or st.session_state.messages[0]["role"] != "system":
            st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            st.session_state.messages[0]["content"] = system_prompt

        # 1) Render history (skip system message in UI)
        for msg in st.session_state.messages[1:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 2) If we have queued input and we are NOT streaming, start streaming phase
        if st.session_state.pending_user_input and not st.session_state.is_streaming:
            # Append user message to history
            st.session_state.messages.append({"role": "user", "content": st.session_state.pending_user_input})

            # Clear queue, mark streaming active
            st.session_state.pending_user_input = None
            st.session_state.is_streaming = True

            # Rerun so the user message appears immediately in the history
            st.rerun()

        # 3) Streaming phase: render assistant bubble ABOVE the input
        if st.session_state.is_streaming:
            with st.chat_message("assistant"):
                if not st.session_state.active_provider:
                    st.error("Please initialize a model first using the settings panel.")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Please initialize a model first using the settings panel."}
                    )
                    st.session_state.is_streaming = False
                    st.rerun()

                provider = st.session_state.providers[st.session_state.active_provider]

                message_placeholder = st.empty()
                full_response = ""

                # Start perf measurement
                st.session_state.performance_visualizer.start_measurement()

                try:
                    for response_chunk in provider.generate(
                        messages=st.session_state.messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                    ):
                        full_response += response_chunk
                        message_placeholder.markdown(full_response + "▌")
                        st.session_state.performance_visualizer.add_token()

                    st.session_state.performance_visualizer.end_measurement(provider.use_kv_cache)
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"Error generating response: {str(e)}"
                    st.error(full_response)

                # Commit assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # End streaming + rerun so input stays last
                st.session_state.is_streaming = False
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # 4) Chat input ALWAYS LAST (this is crucial)
        user_input = st.chat_input("Type your message here...")

        # On submit: queue input and rerun (do NOT stream in this run)
        if user_input and not st.session_state.is_streaming:
            st.session_state.pending_user_input = user_input
            st.rerun()

# ----------------------------
# Tab 2: Performance Comparison
# ----------------------------
with tab2:
    st.header("Performance Comparison")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Performance Settings")
        st.caption("Generate text in Chat to populate metrics.")

        if st.button("Clear Performance Data"):
            st.session_state.performance_visualizer.clear_data()
            st.success("Performance data cleared")

    with col1:
        try:
            st.session_state.performance_visualizer.visualize_performance(st.container())
        except Exception:
            st.info("No performance data available yet. Generate text first.")

# ----------------------------
# Tab 3: Hardware Info
# ----------------------------
with tab3:
    st.header("Hardware Info")

    sys_info = get_system_info()
    rec = get_recommended_config(sys_info)

    st.subheader("Detected System")
    st.json(sys_info)

    st.subheader("Recommended Config")
    st.json(rec)

    # Provider metrics summary (if available)
    st.divider()
    st.subheader("Live Provider Metrics")
    try:
        if st.session_state.active_provider:
            provider = st.session_state.providers[st.session_state.active_provider]
            metrics = provider.get_performance_metrics()
            st.write(metrics)
            if "avg_speed" in metrics:
                st.caption(f"Average generation speed: {metrics['avg_speed']:.2f} tokens/sec")
        else:
            st.caption("Initialize a model to see metrics.")
    except Exception:
        st.caption("Performance metrics not available.")

# ----------------------------
# GitHub badge/link
# ----------------------------
st.markdown(
    """
    <div style="position: fixed; top: 10px; right: 16px; z-index: 1000;">
        <a style="text-decoration:none; font-weight:600;"
            href="https://github.com/Sampreeth-sarma/LLM-Chat"
            target="_blank">
            📂 View on GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
