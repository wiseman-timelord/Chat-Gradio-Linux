# scripts/temporary.py

import time

# Configuration variables with defaults
MODEL_FOLDER = "models"
CONTEXT_SIZE = 8192
BATCH_SIZE = 2048
TEMPERATURE = 0.66
REPEAT_PENALTY = 1.1
MMAP = False
MLOCK = False
MAX_HISTORY_SLOTS = 12
MAX_ATTACH_SLOTS = 6
SESSION_LOG_HEIGHT = 500
INPUT_LINES = 27
CTX_OPTIONS = [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8096, 16384, 32768]
TEMP_OPTIONS = [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0]
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
HISTORY_SLOT_OPTIONS = [4, 8, 10, 12, 16]
ATTACH_SLOT_OPTIONS = [2, 4, 6, 8, 10]
SESSION_LOG_HEIGHT_OPTIONS = [450, 475, 500, 550, 650, 800, 1050, 1300]

# General Constants/Variables
TEMP_DIR = "data/temp"
HISTORY_DIR = "data/history"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
session_label = ""
current_session_id = None
MODELS_LOADED = False
AVAILABLE_MODELS = None
SESSION_ACTIVE = False
MODEL_NAME = "Select_a_model..."
SELECTED_GPU = None
CUDA_VERSION = "Unknown"
COMPUTE_CAPABILITY = "Unknown"
DATA_DIR = None
llm = None
SPEECH_ENABLED = False
AUTO_SUMMARY = False
CURRENT_SUMMARY = ""
VRAM_MB = 0
SYSTEM_RAM_MB = 0  # New: System RAM in MB
DDR_LEVEL = "Unknown"  # New: DDR generation level
LLAMA_CPP_BINARY = "data/llama-cpp/main"  # Constant

# Arrays
session_attached_files = []

# UI Constants
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30

# Options for Dropdowns
ALLOWED_EXTENSIONS = {"bat", "py", "sh", "txt", "json", "yaml"}
MAX_POSSIBLE_HISTORY_SLOTS = 16
MAX_POSSIBLE_ATTACH_SLOTS = 10

# Status text entries
STATUS_TEXTS = {
    "model_loading": "Loading model",
    "model_loaded": "Model ready",
    "model_unloading": "Unloading",
    "model_unloaded": "Model unloaded",
    "vram_calc": "Calculating layers",
    "rag_process": "Analyzing docs",
    "session_restore": "Restoring session",
    "config_saved": "Settings saved",
    "docs_processed": "Docs ready",
    "generating_response": "Generating",
    "response_generated": "Response ready",
    "error": "Error occurred"
}

CHAT_FORMAT_MAP = {
    'qwen2': 'chatml',
    'llama': 'llama-2',
    'qwen3': 'chatml',
    'qwen3moe': 'chatml',
    'deepseek2': 'deepseek',
    'stablelm': 'chatml',
}

# Handling Keywords for Special Model Behaviors
handling_keywords = {
    "code": ["code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell"],
    "uncensored": ["uncensored", "unfiltered", "unbiased", "unlocked"],
    "reasoning": ["reason", "r1", "think"],
    "nsfw": ["nsfw", "adult", "mature", "explicit", "lewd"],
    "roleplay": ["rp", "role", "adventure"]
}

# Prompt template table
current_model_settings = {
    "category": "chat"
}