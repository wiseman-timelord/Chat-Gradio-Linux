# scripts/settings.py

import json
import time
from pathlib import Path
import scripts.temporary as temporary
from scripts.models import get_available_models

CONFIG_PATH = Path("data/persistent.json")

# Default settings
DEFAULTS = {
    "MODEL_FOLDER": "models",
    "CONTEXT_SIZE": 8192,
    "BATCH_SIZE": 2048,
    "TEMPERATURE": 0.66,
    "REPEAT_PENALTY": 1.1,
    "MMAP": False,
    "MLOCK": False,
    "MAX_HISTORY_SLOTS": 12,
    "MAX_ATTACH_SLOTS": 6,
    "SESSION_LOG_HEIGHT": 500,
}

def load_config():
    """Load configuration from persistent.json and set in temporary.py."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {str(e)[:60]}"); time.sleep(1)
            config = {}
        
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
        # Set defaults if JSON doesn't exist
        for key, value in DEFAULTS.items():
            setattr(temporary, key, value)
        temporary.SELECTED_GPU = None
        temporary.CUDA_VERSION = "Unknown"
        temporary.COMPUTE_CAPABILITY = "Unknown"
    
    # Hardcode paths
    temporary.LLAMA_CPP_BINARY = "data/llama-cpp/main"
    
    # Scan for available models
    available_models = get_available_models()
    temporary.AVAILABLE_MODELS = available_models
    temporary.MODEL_NAME = available_models[0] if available_models else "Select_a_model..."
    
    return "Configuration loaded"

def save_config():
    """Save current settings from temporary.py to persistent.json."""
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
    return "Settings saved"