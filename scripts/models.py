# scripts/models.py

import time
import re
import os
from pathlib import Path
from datetime import datetime
import gradio as gr
from scripts.prompts import get_system_message, get_reasoning_instruction
import scripts.temporary as temporary
from scripts.temporary import (
    CONTEXT_SIZE, BATCH_SIZE, LLAMA_CPP_BINARY, SELECTED_GPU,
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED, CHAT_FORMAT_MAP
)

def get_chat_format(metadata):
    """Determine the chat format based on the model's architecture."""
    architecture = metadata.get('general.architecture', 'unknown')
    return CHAT_FORMAT_MAP.get(architecture, 'llama-2')

def get_model_metadata(model_path: str) -> dict:
    """Retrieve metadata from a GGUF model."""
    try:
        from llama_cpp import Llama
        chat_format = 'chatml' if 'qwen' in model_path.lower() else None
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=1,
            verbose=True,
            chat_format=chat_format
        )
        metadata = model.metadata
        del model
        return metadata
    except Exception as e:
        print(f"Metadata error: {str(e)}"); time.sleep(3)
        return {}

def get_available_models():
    """Scan model directory for GGUF files."""
    model_dir = Path(temporary.MODEL_FOLDER)
    files = list(model_dir.glob("*.gguf"))
    models = [f.name for f in files if f.is_file()]
    return models if models else ["Select_a_model..."]

def get_model_settings(model_name):
    """Determine model settings based on name keywords."""
    model_name_lower = model_name.lower()
    is_uncensored = any(keyword in model_name_lower for keyword in temporary.handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in temporary.handling_keywords["reasoning"])
    is_nsfw = any(keyword in model_name_lower for keyword in temporary.handling_keywords["nsfw"])
    is_code = any(keyword in model_name_lower for keyword in temporary.handling_keywords["code"])
    is_roleplay = any(keyword in model_name_lower for keyword in temporary.handling_keywords["roleplay"])
    return {
        "category": "chat",
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "is_nsfw": is_nsfw,
        "is_code": is_code,
        "is_roleplay": is_roleplay,
        "detected_keywords": [kw for kw in temporary.handling_keywords if any(k in model_name_lower for k in temporary.handling_keywords[kw])]
    }

def load_models(model_folder, model, llm_state, models_loaded_state):
    """Load a GGUF model with CUDA and unified memory."""
    from scripts.settings import save_config

    save_config()

    if model in ["Select_a_model...", "No models found"]:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    metadata = get_model_metadata(str(model_path))
    chat_format = get_chat_format(metadata)

    try:
        from llama_cpp import Llama
    except ImportError:
        error_msg = "Error: llama-cpp-python not installed."
        log_error(error_msg)
        return error_msg, False, llm_state, models_loaded_state

    try:
        if models_loaded_state:
            unload_models(llm_state, models_loaded_state)

        # Set CUDA device
        if temporary.SELECTED_GPU is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(temporary.SELECTED_GPU)
        else:
            error_msg = "Error: No GPU selected in configuration."
            log_error(error_msg)
            return error_msg, False, llm_state, models_loaded_state

        new_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.CONTEXT_SIZE,
            n_batch=temporary.BATCH_SIZE,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=True,
            chat_format=chat_format
            # Unified memory enabled via compilation flag -DLLAMA_CUDA_UNIFIED_MEMORY=ON
        )

        # Test inference
        test_output = new_llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=temporary.BATCH_SIZE,
            stream=False
        )
        print(f"Test inference successful: {test_output}")

        temporary.MODEL_NAME = model
        status = f"Model '{model}' loaded with chat_format '{chat_format}' on GPU {temporary.SELECTED_GPU}."
        return status, True, new_llm, True
    except RuntimeError as e:
        error_msg = f"CUDA error loading model: {str(e)}. Check VRAM (reduce n_batch) or CUDA setup."
        rint(f"GPU error: {error_msg}"); time.sleep(3)
        return error_msg, False, None, False
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"Load error: {error_msg}"); time.sleep(3)
        return error_msg, False, None, False

def unload_models(llm_state, models_loaded_state):
    """Unload the current model."""
    import gc
    if models_loaded_state:
        del llm_state
        gc.collect()
        return "Model unloaded successfully.", None, False
    return "No model loaded to unload.", llm_state, models_loaded_state

def get_response_stream(session_log, settings, web_search_enabled=False, search_results=None, cancel_event=None, llm_state=None, models_loaded_state=False):
    """Generate a response stream using the loaded model."""
    import re

    def should_stream(input_text, settings):
        stream_keywords = ["write", "generate", "story", "report", "essay", "explain", "describe"]
        input_length = len(input_text.strip())
        is_long_input = input_length > 100
        is_creative_task = any(keyword in input_text.lower() for keyword in stream_keywords)
        is_interactive_mode = settings.get("is_reasoning", False)
        return is_creative_task or is_long_input or (is_interactive_mode and input_length > 50)

    if not models_loaded_state or llm_state is None:
        yield "Error: No model loaded."
        return

    n_ctx = temporary.CONTEXT_SIZE
    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False)
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."

    if web_search_enabled and search_results:
        system_message += "\n\nSearch Results:\n" + re.sub(r'https?://(www\.)?([^/]+).*', r'\2', str(search_results))

    try:
        system_tokens = len(llm_state.tokenize(system_message.encode('utf-8')))
    except Exception as e:
        error_msg = f"Error tokenizing system message: {str(e)}"
        print(f"Stream error: {error_msg}"); time.sleep(3)
        yield error_msg
        return

    if not session_log or len(session_log) < 2 or session_log[-2]['role'] != 'user':
        yield "Error: No user input to process."
        return

    user_query = session_log[-2]['content'].replace("User:\n", "", 1).strip()
    try:
        user_tokens = len(llm_state.tokenize(user_query.encode('utf-8')))
    except Exception as e:
        error_msg = f"Error tokenizing user query: {str(e)}"
        print(f"Stream error: {error_msg}"); time.sleep(3)
        yield error_msg
        return

    available_tokens = n_ctx - system_tokens - user_tokens - (temporary.BATCH_SIZE // 8)
    if available_tokens < 0:
        yield "Error: Context size exceeded. Reduce context_size or n_batch."
        return

    messages = [{"role": "system", "content": system_message}]
    current_tokens = 0
    
    for msg in reversed(session_log[:-2]):
        content = msg['content'].strip()
        if msg['role'] == 'user':
            content = content.replace("User:\n", "", 1)
        try:
            msg_tokens = len(llm_state.tokenize(content.encode('utf-8')))
            if current_tokens + msg_tokens > available_tokens:
                break
            messages.insert(1, {"role": msg['role'], "content": content})
            current_tokens += msg_tokens
        except Exception:
            continue

    messages.append({"role": "user", "content": user_query})

    try:
        if should_stream(user_query, settings):
            for chunk in llm_state.create_chat_completion(
                messages=messages,
                max_tokens=temporary.BATCH_SIZE,
                temperature=float(settings.get("temperature", temporary.TEMPERATURE)),
                repeat_penalty=float(settings.get("repeat_penalty", temporary.REPEAT_PENALTY)),
                stream=True
            ):
                if cancel_event and cancel_event.is_set():
                    yield "<CANCELLED>"
                    return
                if chunk.get('choices'):
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    if content:
                        content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
                        content = re.sub(r'\n{2,}', '\n', content)
                        yield content
        else:
            response = llm_state.create_chat_completion(
                messages=messages,
                max_tokens=temporary.BATCH_SIZE,
                stream=False
            )
            content = response['choices'][0]['message']['content']
            content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\n{2,}', '\n', content)
            yield content
    except RuntimeError as e:
        error_msg = f"CUDA error during inference: {str(e)}. Try reducing n_batch."
        print(f"Runtime error: {error_msg}"); time.sleep(3)
        yield error_msg
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"Response error: {error_msg}"); time.sleep(3)
        yield error_msg