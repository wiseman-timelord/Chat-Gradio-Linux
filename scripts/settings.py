# scripts/settings.py

import json
from pathlib import Path
import scripts.temporary as temporary
from scripts.models import get_available_models

CONFIG_PATH = Path("data/persistent.json")

# Default settings
DEFAULTS = {
    "MODEL_FOLDER": "models",
    "CONTEXT_SIZE": 8192,
    "BATCH_SIZE": 2048,  # Overridden by installer based on VRAM
    "TEMPERATURE": 0.66,
    "REPEAT_PENALTY": 1.1,
    "MMAP": False,
    "MLOCK": False,
    "MAX_HISTORY_SLOTS": 12,
    "MAX_ATTACH_SLOTS": 6,
    "SESSION_LOG_HEIGHT": 500,
    "VRAM_OPTIONS": [2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536],
    "CTX_OPTIONS": [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072],
    "BATCH_OPTIONS": [128, 256, 512, 1024, 2048, 4096, 8096, 16384, 32768],
    "TEMP_OPTIONS": [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0],
    "REPEAT_OPTIONS": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "HISTORY_SLOT_OPTIONS": [4, 8, 10, 12, 16],
    "ATTACH_SLOT_OPTIONS": [2, 4, 6, 8, 10],
    "SESSION_LOG_HEIGHT_OPTIONS": [450, 475, 500, 550, 650, 800, 1050, 1300],
}

def load_config():
    """
    Load configuration from persistent.json and set in temporary.py.
    Always scan for available models and cache the result.
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid persistent.json: {str(e)}")
        
        # Load model_settings
        model_settings = config.get("model_settings", {})
        temporary.MODEL_FOLDER = model_settings.get("model_dir", DEFAULTS["MODEL_FOLDER"])
        temporary.CONTEXT_SIZE = model_settings.get("context_size", DEFAULTS["CONTEXT_SIZE"])
        temporary.TEMPERATURE = model_settings.get("temperature", DEFAULTS["TEMPERATURE"])
        temporary.REPEAT_PENALTY = model_settings.get("repeat_penalty", DEFAULTS["REPEAT_PENALTY"])
        temporary.MMAP = model_settings.get("mmap", DEFAULTS["MMAP"])
        temporary.MLOCK = model_settings.get("mlock", DEFAULTS["MLOCK"])
        temporary.BATCH_SIZE = model_settings.get("n_batch", DEFAULTS["BATCH_SIZE"])
        temporary.MAX_HISTORY_SLOTS = model_settings.get("max_history_slots", DEFAULTS["MAX_HISTORY_SLOTS"])
        temporary.MAX_ATTACH_SLOTS = model_settings.get("max_attach_slots", DEFAULTS["MAX_ATTACH_SLOTS"])
        temporary.SESSION_LOG_HEIGHT = model_settings.get("session_log_height", DEFAULTS["SESSION_LOG_HEIGHT"])
        
        # Load backend_config
        backend_config = config.get("backend_config", {})
        temporary.SELECTED_GPU = backend_config.get("selected_gpu", None)
        temporary.CUDA_VERSION = backend_config.get("cuda_version", "Unknown")
        temporary.COMPUTE_CAPABILITY = backend_config.get("compute_capability", "Unknown")
        
    else:
        # Set defaults if JSON doesn’t exist
        for key, value in DEFAULTS.items():
            setattr(temporary, key, value)
        temporary.SELECTED_GPU = None
        temporary.CUDA_VERSION = "Unknown"
        temporary.COMPUTE_CAPABILITY = "Unknown"
        temporary.MODEL_NAME = "Select_a_model..."
    
    # Hardcode the llama.cpp binary path
    temporary.LLAMA_CPP_BINARY = "data/llama-cpp/main"
    
    # Scan for available models and cache the result
    available_models = get_available_models()
    temporary.AVAILABLE_MODELS = available_models
    
    # Validate model_name against available models
    if CONFIG_PATH.exists():
        model_settings = config.get("model_settings", {})
        if "model_name" in model_settings and model_settings["model_name"] in available_models:
            temporary.MODEL_NAME = model_settings["model_name"]
        else:
            temporary.MODEL_NAME = available_models[0] if available_models else "Select_a_model..."
    else:
        temporary.MODEL_NAME = "Select_a_model..."
    
    return "Configuration loaded."

def save_config():
    """
    Save current settings from temporary.py to persistent.json.
    """
    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "model_name": temporary.MODEL_NAME,
            "context_size": temporary.CONTEXT_SIZE,
            "temperature": temporary.TEMPERATURE,
            "repeat_penalty": temporary.REPEAT_PENALTY,
            "mmap": temporary.MMAP,
            "mlock": temporary.MLOCK,
            "n_batch": temporary.BATCH_SIZE,
            "max_history_slots": temporary.MAX_HISTORY_SLOTS,
            "max_attach_slots": temporary.MAX_ATTACH_SLOTS,
            "session_log_height": temporary.SESSION_LOG_HEIGHT,
        },
        "backend_config": {
            "selected_gpu": temporary.SELECTED_GPU,
            "cuda_version": temporary.CUDA_VERSION,
            "compute_capability": temporary.COMPUTE_CAPABILITY,
        }
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    return "Settings saved."