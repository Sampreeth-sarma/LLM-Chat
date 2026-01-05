import torch
import platform
import psutil
import GPUtil
import os

def get_system_info():
    """Get system information including CPU, RAM, and GPU if available"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {
            "processor": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
            "available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
        },
        "gpu": None
    }
    
    # Check for GPUs
    try:
        if torch.cuda.is_available():
            info["gpu"] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "name": torch.cuda.get_device_name(i),
                    "index": i,
                    "memory_total": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),  # GB
                }
                try:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_info["memory_used"] = round(gpus[i].memoryUsed / 1024, 2)  # GB
                        gpu_info["memory_free"] = round(gpus[i].memoryFree / 1024, 2)  # GB
                        gpu_info["load"] = round(gpus[i].load * 100, 2)  # %
                except:
                    pass
                info["gpu"].append(gpu_info)
    except:
        pass
        
    return info

def get_recommended_config(info=None):
    """Recommend configuration based on available hardware"""
    if not info:
        info = get_system_info()
    
    config = {
        "provider": "ollama",  # Default to Ollama for CPU
        "model_size": "small",
        "tensor_parallel": 1,
        "kv_cache_precision": "auto"
    }
    
    # If GPU available, recommend vLLM
    if info["gpu"]:
        config["provider"] = "vllm"
        
        # Calculate total available GPU memory across all GPUs
        total_gpu_mem = sum(gpu.get("memory_total", 0) for gpu in info["gpu"])
        
        # Recommend model size based on available GPU memory
        if total_gpu_mem > 40:  # More than 40GB
            config["model_size"] = "large"
            config["tensor_parallel"] = min(4, len(info["gpu"]))
        elif total_gpu_mem > 20:  # More than 20GB
            config["model_size"] = "medium"
            config["tensor_parallel"] = min(2, len(info["gpu"]))
        else:
            config["model_size"] = "small"
            
    # Set quantization recommendation
    if config["provider"] == "vllm" and info["gpu"]:
        if any("RTX" in gpu.get("name", "") for gpu in info["gpu"]):
            config["kv_cache_precision"] = "fp16"  # For newer NVIDIA GPUs
        else:
            config["kv_cache_precision"] = "int8"  # For older or limited VRAM GPUs
            
    return config