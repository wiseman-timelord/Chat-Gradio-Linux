# scripts/settings.py

import json, time
from pathlib import Path
import os
import scripts.temporary as temporary
from scripts.models import get_available_models

CONFIG_PATH = temporary.CONFIG_PATH

def load_config():
    """Load configuration from persistent.json and set in temporary.py."""
    config_path = Path(temporary.CONFIG_PATH)
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {str(e)[:60]}"); time.sleep(1)
            config = {}
        
        # Load model_settings
        model_settings = config.get("model_settings", {})
        temporary.MODEL_FOLDER = model_settings.get("model_dir", temporary.DEFAULTS["MODEL_FOLDER"])
        temporary.CONTEXT_SIZE = model_settings.get("context_size", temporary.DEFAULTS["CONTEXT_SIZE"])
        temporary.TEMPERATURE = model_settings.get("temperature", temporary.DEFAULTS["TEMPERATURE"])
        temporary.REPEAT_PENALTY = model_settings.get("repeat_penalty", temporary.DEFAULTS["REPEAT_PENALTY"])
        temporary.MMAP = model_settings.get("mmap", temporary.DEFAULTS["MMAP"])
        temporary.MLOCK = model_settings.get("mlock", temporary.DEFAULTS["MLOCK"])
        temporary.BATCH_SIZE = model_settings.get("n_batch", temporary.DEFAULTS["BATCH_SIZE"])
        temporary.MAX_HISTORY_SLOTS = model_settings.get("max_history_slots", temporary.DEFAULTS["MAX_HISTORY_SLOTS"])
        temporary.MAX_ATTACH_SLOTS = model_settings.get("max_attach_slots", temporary.DEFAULTS["MAX_ATTACH_SLOTS"])
        temporary.SESSION_LOG_HEIGHT = model_settings.get("session_log_height", temporary.DEFAULTS["SESSION_LOG_HEIGHT"])
        
        # Load backend_config
        backend_config = config.get("backend_config", {})
        temporary.SELECTED_GPU = backend_config.get("selected_gpu", None)
        temporary.CUDA_VERSION = backend_config.get("cuda_version", "Unknown")
        temporary.COMPUTE_CAPABILITY = backend_config.get("compute_capability", "Unknown")
        temporary.VRAM_MB = backend_config.get("vram_mb", 0)
        
        # Note: SYSTEM_RAM_MB and DDR_LEVEL are detected live and not loaded from config
    else:
        # Set defaults if JSON doesn't exist
        for key, value in temporary.DEFAULTS.items():
            setattr(temporary, key, value)
        temporary.SELECTED_GPU = None
        temporary.CUDA_VERSION = "Unknown"
        temporary.COMPUTE_CAPABILITY = "Unknown"
        temporary.VRAM_MB = 0
        temporary.SYSTEM_RAM_MB = 0  # Will be detected live
        temporary.DDR_LEVEL = "Unknown"  # Will be detected live
    
    # Hardcode paths
    temporary.LLAMA_CPP_BINARY = str(Path(temporary.DATA_DIR) / "llama-cpp" / "main")
    
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
            "vram_mb": temporary.VRAM_MB,
            # Note: We intentionally don't save SYSTEM_RAM_MB and DDR_LEVEL
            # as they are system-dependent and should be detected live
        }
    }
    
    config_path = Path(temporary.CONFIG_PATH)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return "Settings saved"
    except Exception as e:
        print(f"Save error: {str(e)[:60]}"); time.sleep(3)
        return f"Save failed: {str(e)[:60]}"