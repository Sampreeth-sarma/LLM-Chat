import streamlit as st
import torch
import time
import pandas as pd
from model_providers.ollama import OllamaProvider
from model_providers.vllm import VLLMProvider
from utils.prompt_manager import PromptManager
from utils.device_utils import get_system_info, get_recommended_config
from utils.visualization import PerformanceVisualizer

# Page configuration
st.set_page_config(page_title="LLM Demo", page_icon="🤖", layout="wide")

# CSS fixes for chat interface
st.markdown("""
    <span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;style&gt;</span><span style="color: black; font-weight: normal;">
    .stChatFloatingInputContainer {
        display: flex !important;
        padding-bottom: 20px !important;
        visibility: visible !important;
    }
    .stChatInputContainer {
        display: flex !important;
        visibility: visible !important;
    }
    </span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/style&gt;</span><br><br></span>
    """, unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "providers" not in st.session_state:
    st.session_state.providers = {}
    
if "active_provider" not in st.session_state:
    st.session_state.active_provider = None
    
if "prompt_manager" not in st.session_state:
    st.session_state.prompt_manager = PromptManager()
    
if "performance_visualizer" not in st.session_state:
    st.session_state.performance_visualizer = PerformanceVisualizer()

# UI layout with tabs
st.title("🤖 LLM Chat Application Demo")

tab1, tab2, tab3 = st.tabs(["Chat", "Performance Comparison", "Hardware Info"])

with tab1:
    # Main chat interface with sidebar for settings
    col1, col2 = st.columns([3, 1])

    with col2:
        st.header("Model Settings")
        
        # Model Provider Selection
        provider_option = st.selectbox(
            "Model Provider",
            ["Ollama", "vLLM"],
            help="Select the model provider"
        )
        
        # Model Selection based on provider
        if provider_option == "Ollama":
            try:
                ollama_provider = OllamaProvider()
                available_models = ollama_provider.list_models()
            except:
                available_models = ["llama2", "llama3", "mistral", "qwen", "deepseek-coder"]
                
            selected_model = st.selectbox("Model", available_models)
            
        elif provider_option == "vLLM":
            try:
                temp_provider = VLLMProvider("meta-llama/Llama-2-7b-chat-hf")
                available_models = temp_provider.list_models()
            except:
                available_models = [
                    "meta-llama/Llama-2-7b-chat-hf",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "Qwen/Qwen-7B-Chat"
                ]
            selected_model = st.selectbox("Model", available_models)
        
        # KV Cache Toggle - Featured prominently as it's a key demo point
        kv_cache_enabled = st.toggle("Enable KV Cache", value=True,
                               help="KV Cache stores past key-value pairs to speed up generation")
        
        # Model family detection for appropriate prompt formatting
        if "llama" in selected_model.lower():
            model_family = "llama"
        elif "mistral" in selected_model.lower():
            model_family = "mistral"
        elif "qwen" in selected_model.lower():
            model_family = "qwen"
        elif "deepseek" in selected_model.lower():
            model_family = "deepseek"
        else:
            model_family = "generic"
            
        st.caption(f"Detected model family: {model_family}")
            
        # System prompt selection
        st.subheader("System Prompt")
        template_options = st.session_state.prompt_manager.list_templates()
        selected_template = st.selectbox("Select Template", template_options)
        
        system_prompt = st.text_area(
            "Customize System Prompt",
            value=st.session_state.prompt_manager.get_template(selected_template),
            height=100
        )
        
        # Advanced settings in an expander
        with st.expander("Generation Parameters"):
            # Temperature slider
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
            
            col1_inner, col2_inner = st.columns(2)
            with col1_inner:
                # Top-p (nucleus sampling)
                top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
            
            with col2_inner:
                # Top-k
                top_k = st.slider("Top-k", min_value=1, max_value=100, value=40, step=1)
            
            # Max tokens
            max_tokens = st.number_input("Max Tokens", min_value=10, max_value=4096, value=2048, step=10)
            
            # Provider-specific settings
            if provider_option == "vLLM":
                # Attention method selection
                try:
                    temp_provider = VLLMProvider("meta-llama/Llama-2-7b-chat-hf")
                    attention_methods = temp_provider.list_attention_methods()
                except:
                    attention_methods = ["auto", "flash_attention", "flash_attention_2", "eager"]
                
                attention_method = st.selectbox(
                    "Attention Method", 
                    attention_methods,
                    help="Different attention implementations (Flash Attention is fastest on modern GPUs)"
                )
                
                # GPU selection
                if torch.cuda.is_available():
                    available_gpus = list(range(torch.cuda.device_count()))
                    selected_gpu = st.selectbox(
                        "GPU", 
                        available_gpus,
                        format_func=lambda x: f"GPU {x}: {torch.cuda.get_device_name(x)}"
                    )
                    
                    # Tensor parallel size
                    tensor_parallel = st.slider(
                        "Tensor Parallel Size", 
                        min_value=1, 
                        max_value=len(available_gpus),
                        value=1
                    )
                else:
                    st.warning("No CUDA-capable GPUs detected")
                    selected_gpu = None
                    tensor_parallel = 1
        
        # Initialize model button
        if st.button("Initialize Model"):
            with st.spinner(f"Initializing {selected_model}..."):
                try:
                    provider_key = f"{provider_option}_{selected_model}"
                    
                    if provider_option == "Ollama":
                        provider = OllamaProvider(model_name=selected_model)
                        
                    elif provider_option == "vLLM":
                        provider = VLLMProvider(
                            model_name=selected_model,
                            gpu_id=selected_gpu if torch.cuda.is_available() else None,
                            tensor_parallel_size=tensor_parallel if torch.cuda.is_available() else 1,
                            attention_method=attention_method
                        )
                    
                    # Set KV cache state
                    provider.set_kv_cache(kv_cache_enabled)
                    
                    # Store and activate the provider
                    st.session_state.providers[provider_key] = provider
                    st.session_state.active_provider = provider_key
                    st.success(f"Model {selected_model} initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Error initializing model: {str(e)}")
                    
        # Update KV cache for active provider
        if st.session_state.active_provider and st.session_state.active_provider in st.session_state.providers:
            active_provider = st.session_state.providers[st.session_state.active_provider]
            if active_provider.use_kv_cache != kv_cache_enabled:
                active_provider.set_kv_cache(kv_cache_enabled)
                st.info(f"KV Cache {'enabled' if kv_cache_enabled else 'disabled'}")

    # FIXED CHAT INTERFACE
    with col1:
        # Create a container for the chat interface
        chat_container = st.container()
        
        with chat_container:
            # Add system message to chat if not already there
            if not st.session_state.messages or st.session_state.messages[0]["role"] != "system":
                st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                # Update the system message if it's different
                if st.session_state.messages[0]["content"] != system_prompt:
                    st.session_state.messages[0]["content"] = system_prompt
            
            # Display welcome message if no messages
            if len(st.session_state.messages) <= 1:  # Only system message present
                st.info("👋 Send a message to start chatting with the LLM!")
            
            # Chat history display (skip system message)
            for message in st.session_state.messages[1:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Input and response (outside the chat_container to avoid rendering issues)
        user_input = st.chat_input("Type your message here...", key="unique_chat_input")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Force a rerun to display the user message before generating response
            st.rerun()

        # Handle response generation (this runs after the rerun)
        # Check if the last message is from the user and needs a response
        if (st.session_state.messages and 
            len(st.session_state.messages) > 1 and 
            st.session_state.messages[-1]["role"] == "user"):
            
            user_message = st.session_state.messages[-1]["content"]
            
            # Display the existing user message
            with st.chat_message("user"):
                st.markdown(user_message)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                if not st.session_state.active_provider:
                    st.error("Please initialize a model first using the settings panel.")
                else:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    provider = st.session_state.providers[st.session_state.active_provider]
                    
                    # Start performance measurement
                    st.session_state.performance_visualizer.start_measurement()
                    
                    # Stream the response
                    try:
                        for response_chunk in provider.generate(
                            messages=st.session_state.messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            top_k=top_k
                        ):
                            full_response += response_chunk
                            message_placeholder.markdown(full_response + "▌")
                            st.session_state.performance_visualizer.add_token()
                        
                        # End performance measurement
                        st.session_state.performance_visualizer.end_measurement(provider.use_kv_cache)
                        
                        # Final response without cursor
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        full_response = f"Error: {str(e)}"
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Clear chat button
        if len(st.session_state.messages) > 1 and st.button("Clear Chat History"):
            system_message = st.session_state.messages[0] if st.session_state.messages else {"role": "system", "content": system_prompt}
            st.session_state.messages = [system_message]
            st.rerun()

with tab2:
    st.header("Performance Comparison")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Performance Settings")
        
        if st.button("Run Performance Test", help="Runs a standardized test to compare performance with and without KV cache"):
            st.info("Running performance test...")
            # This would be implemented with actual test logic
            
        if st.button("Clear Performance Data"):
            st.session_state.performance_visualizer.clear_data()
            st.success("Performance data cleared")
    
    with col1:
        # Use try/except in case visualize_performance is not defined yet
        try:
            st.session_state.performance_visualizer.visualize_performance(st.container())
        except:
            st.info("No performance data available yet. Generate text first or run a performance test.")
    
    st.subheader("What is KV Caching?")
    st.markdown("""
    **Key-Value Caching** is an optimization technique used in transformer-based language models to avoid redundant computation:
    
    - **Without KV Cache**: The model recalculates attention for all previous tokens on each new token generation
    - **With KV Cache**: The model stores previously computed key-value pairs, reusing them for subsequent tokens
    
    **Benefits:**
    1. Significantly faster generation (typically 2-4x speedup)
    2. Reduced computational overhead
    3. Lower memory bandwidth usage
    
    **Trade-offs:**
    1. Increased memory usage (storing the KV cache)
    2. Can limit context length for very long sequences
    """)
    
    st.subheader("Different Attention Implementation Methods")
    st.markdown("""
    Modern LLMs use various attention implementation strategies:
    
    - **Standard Attention**: The original implementation - accurate but slower
    - **Flash Attention**: Optimized implementation that maximizes GPU utilization
    - **Flash Attention 2**: Improved version with better performance
    - **Group Query Attention (GQA)**: Reduces computation by sharing keys and values across multiple query heads
    - **Multi-Query Attention (MQA)**: Similar to GQA but more aggressive sharing
    - **Sliding Window Attention**: Restricts attention to a local window, reducing complexity
    
    These optimizations enable running larger models more efficiently.
    """)

with tab3:
    st.header("System Hardware Information")
    
    try:
        system_info = get_system_info()
    except:
        system_info = {
            "platform": "Information not available",
            "python_version": "Information not available",
            "cpu": {"processor": "Information not available", "cores": 0, "threads": 0},
            "memory": {"available": 0, "total": 0},
            "gpu": None
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
        if system_info['gpu']:
            for i, gpu in enumerate(system_info['gpu']):
                st.write(f"**GPU {i}:** {gpu['name']}")
                if 'memory_free' in gpu:
                    st.write(f"   - Memory: {gpu['memory_free']}GB free / {gpu['memory_total']}GB total")
                if 'load' in gpu:
                    st.write(f"   - Load: {gpu['load']}%")
                    
                    # Draw a progress bar for GPU utilization
                    st.progress(gpu['load'] / 100)
        else:
            st.write("**GPU:** None detected")
    
    # Recommended configurations based on hardware
    st.subheader("Recommended Configurations")
    
    try:
        recommended_config = get_recommended_config()
        st.json(recommended_config)
    except:
        st.error("Could not generate recommended configurations")
    
    st.markdown("""
    ### Model Size Guidelines
    
    | Model Type | Parameters | Min VRAM | Recommended VRAM |
    |------------|------------|----------|------------------|
    | Small      | 7B-8B      | 8GB      | 12GB             |
    | Medium     | 13B-14B    | 16GB     | 24GB             |
    | Large      | 30B-70B    | 32GB     | 48GB+            |
    """)

# Footer and additional info
st.markdown("---")
st.caption("Built with Streamlit. Created for demonstration purposes.")

if st.session_state.active_provider:
    try:
        provider = st.session_state.providers[st.session_state.active_provider]
        metrics = provider.get_performance_metrics()
        st.caption(f"Average generation speed: {metrics['avg_speed']:.2f} tokens/sec")
    except:
        st.caption("Performance metrics not available")

# Add GitHub-style corner badge
st.markdown(
    """
    <span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;style&gt;</span><span style="color: black; font-weight: normal;">
    .github-corner:hover .octo-arm {
        animation: octocat-wave 560ms ease-in-out;
    }
    @keyframes octocat-wave {
        0%, 100% { transform: rotate(0); }
        20%, 60% { transform: rotate(-25deg); }
        40%, 80% { transform: rotate(10deg); }
    }
    @media (max-width: 500px) {
        .github-corner:hover .octo-arm {
            animation: none;
        }
        .github-corner .octo-arm {
            animation: octocat-wave 560ms ease-in-out;
        }
    }
    </span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/style&gt;</span><br><br></span>
    <a aria-label="View source on GitHub" class="github-corner" href="https://github.com/yourusername/llm-chat-demo">
        <span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;svg&gt;</span><span style="color: black; font-weight: normal;">
            
            
            
        </span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/svg&gt;</span><br><br></span>
    </a>
    """,
    unsafe_allow_html=True,
)