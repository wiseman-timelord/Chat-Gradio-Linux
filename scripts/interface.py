# scripts/interface.py

import gradio as gr
import re
import os
import json
import pyperclip
import asyncio
import queue
import threading
import time
import random
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import scripts.temporary as temporary
import scripts.settings as settings
from scripts.settings import save_config
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    ALLOWED_EXTENSIONS, SELECTED_GPU, SESSION_ACTIVE,
    HISTORY_DIR, MODEL_NAME, STATUS_TEXTS
)
from scripts import utility
from scripts.utility import (
    web_search, get_saved_sessions, load_session_history, save_session_history
)
from scripts.models import (
    get_response_stream, get_available_models, unload_models, get_model_settings, load_models
)

def set_loading_status():
    return "Loading model..."

def get_panel_choices(model_settings):
    choices = ["History", "Attach"]
    if model_settings.get("is_nsfw", False) or model_settings.get("is_roleplay", False):
        if "Attach" in choices:
            choices.remove("Attach")
    return choices

def update_panel_choices(model_settings, current_panel):
    choices = get_panel_choices(model_settings)
    if current_panel not in choices:
        current_panel = choices[0] if choices else "History"
    return gr.update(choices=choices, value=current_panel), current_panel

def generate_summary(last_response, llm_state):
    if not last_response:
        return "No response to summarize."
    summary_prompt = f"Summarize the following response in under 256 characters:\n\n{last_response}"
    try:
        response = llm_state.create_chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=256,
            temperature=0.5,
            stream=False
        )
        summary = response['choices'][0]['message']['content'].strip()
        if len(summary) > 256:
            summary = summary[:253] + "..."
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def process_attach_files(files, attached_files, models_loaded):
    if not models_loaded:
        return "Error: Load model first.", attached_files
    return utility.process_files(files, attached_files, temporary.MAX_ATTACH_SLOTS, is_attach=True)

def update_config_settings(ctx, batch, temp, repeat, model):
    temporary.CONTEXT_SIZE = int(ctx)
    temporary.BATCH_SIZE = int(batch)
    temporary.TEMPERATURE = float(temp)
    temporary.REPEAT_PENALTY = float(repeat)
    temporary.MODEL_NAME = model
    return f"Updated settings: Context={ctx}, Batch={batch}, Temp={temp}, Repeat={repeat}, Model={model}"

def save_all_settings():
    settings.save_config()
    return "Settings saved successfully."

def set_session_log_base_height(new_height):
    temporary.SESSION_LOG_HEIGHT = int(new_height)
    return gr.update(height=temporary.SESSION_LOG_HEIGHT)

def estimate_lines(text, chars_per_line=80):
    if not text:
        return 0
    segments = text.split('\n')
    total_lines = 0
    for segment in segments:
        total_lines += max(1, (len(segment) + chars_per_line - 1) // chars_per_line)
    return total_lines

def update_session_log_height(text):
    lines = estimate_lines(text)
    initial_lines = 3
    max_lines = temporary.USER_INPUT_MAX_LINES
    adjustment = 0 if lines <= initial_lines else min(lines - initial_lines, max_lines - initial_lines) * 20
    new_height = max(100, temporary.SESSION_LOG_HEIGHT - adjustment)
    return gr.update(height=new_height)

def get_initial_model_value():
    available_models = temporary.AVAILABLE_MODELS or get_available_models()
    base_choices = ["Select_a_model..."]
    if available_models != ["Select_a_model..."]:
        available_models = base_choices + [m for m in available_models if m not in base_choices]
    else:
        available_models = base_choices
    default_model = temporary.MODEL_NAME if temporary.MODEL_NAME in available_models and temporary.MODEL_NAME not in base_choices else (available_models[1] if len(available_models) > 1 else "Select_a_model...")
    is_reasoning = get_model_settings(default_model)["is_reasoning"] if default_model not in base_choices else False
    return default_model, is_reasoning

def update_model_list(new_dir):
    temporary.MODEL_FOLDER = new_dir
    choices = get_available_models()
    value = choices[0] if choices and choices[0] != "Select_a_model..." else "Select_a_model..."
    return gr.update(choices=choices, value=value)

def handle_model_selection(model_name, model_folder_state):
    if not model_name:
        return model_folder_state, model_name, "No model selected."
    return model_folder_state, model_name, f"Selected model: {model_name}"

def browse_on_click(current_path):
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(initialdir=current_path or os.path.expanduser("~"))
    root.destroy()
    return folder_selected if folder_selected else current_path

def start_new_session(models_loaded):
    if not models_loaded:
        return [], "Load model first on Configuration page...", gr.update(interactive=False), gr.update()
    temporary.current_session_id = None
    temporary.session_label = ""
    temporary.SESSION_ACTIVE = True
    return [], "Type input and click Send to begin...", gr.update(interactive=True), gr.update()

def load_session_by_index(index):
    sessions = utility.get_saved_sessions()
    if index < len(sessions):
        session_file = sessions[index]
        session_id, label, history, attached_files = utility.load_session_history(Path(HISTORY_DIR) / session_file)
        temporary.current_session_id = session_id
        temporary.session_label = label
        temporary.SESSION_ACTIVE = True
        return history, attached_files, f"Loaded session: {label}"
    return [], [], "No session to load"

def copy_last_response(session_log):
    if session_log and session_log[-1]['role'] == 'assistant':
        response = session_log[-1]['content']
        clean_response = re.sub(r'<[^>]+>', '', response)
        pyperclip.copy(clean_response)
        return "AI Response copied to clipboard."
    return "No response available to copy."

def shutdown_program(llm_state, models_loaded_state):
    if models_loaded_state:
        unload_models(llm_state, models_loaded_state)
    demo.close()
    os._exit(0)

def update_file_slot_ui(file_list, is_attach=True):
    button_updates = []
    max_slots = temporary.MAX_POSSIBLE_ATTACH_SLOTS
    for i in range(max_slots):
        if i < len(file_list):
            filename = Path(file_list[i]).name
            short_name = filename[:36] + ".." if len(filename) > 38 else filename
            label = short_name
            visible = True
        else:
            label = ""
            visible = False
        button_updates.append(gr.update(value=label, visible=visible, variant="primary"))
    visible = len(file_list) < temporary.MAX_ATTACH_SLOTS if is_attach else True
    return button_updates + [gr.update(visible=visible)]

def update_session_buttons():
    sessions = utility.get_saved_sessions()[:temporary.MAX_HISTORY_SLOTS]
    button_updates = []
    for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS):
        if i < len(sessions):
            session_path = Path(HISTORY_DIR) / sessions[i]
            stat = session_path.stat()
            update_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            session_id, label, _, _ = utility.load_session_history(session_path)
            btn_label = f"{update_time} - {label}"
            visible = True
        else:
            btn_label = ""
            visible = False
        button_updates.append(gr.update(value=btn_label, visible=visible))
    return button_updates

def update_action_button(phase):
    if phase == "waiting_for_input":
        return gr.update(value="Send Input", variant="secondary", elem_classes=["send-button-green"], interactive=True)
    elif phase == "afterthought_countdown":
        return gr.update(value="Cancel Submission", variant="secondary", elem_classes=["send-button-orange"], interactive=False)
    elif phase == "generating_response":
        return gr.update(value="Wait For Response", variant="secondary", elem_classes=["send-button-red"], interactive=True)
    else:
        return gr.update(value="Unknown Phase", variant="secondary", elem_classes=["send-button-green"], interactive=False)

async def conversation_interface(
    user_input, session_log, loaded_files, is_reasoning_model, cancel_flag,
    web_search_enabled, interaction_phase, llm_state, models_loaded_state,
    summary_enabled
):
    if not models_loaded_state or not llm_state:
        yield session_log, "Please load a model first.", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        return

    if not user_input.strip():
        yield session_log, "No input provided.", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        return

    original_input = user_input
    if temporary.session_attached_files:
        for file in temporary.session_attached_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    user_input += f"\n\nAttached File ({Path(file).name}):\n{f.read()}"
            except Exception as e:
                print(f"Error reading file {file}: {e}")

    session_log.append({'role': 'user', 'content': f"User:\n{user_input}"})
    session_log.append({'role': 'assistant', 'content': "AI-Chat:\n"})
    interaction_phase = "afterthought_countdown"
    
    yield session_log, "Processing...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(interactive=False), gr.update(), gr.update()

    input_length = len(original_input.strip())
    countdown_seconds = 1 if input_length <= 25 else 3 if input_length <= 100 else 5
    progress_indicators = ["⏳", "⌛", "⏰", "⌚", "⏲", "⏱", "🕰"]
    
    for i in range(countdown_seconds, -1, -1):
        current_progress = random.choice(progress_indicators)
        yield session_log, f"{current_progress} Afterthought countdown... {i}s", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        await asyncio.sleep(1)
        if cancel_flag:
            session_log.pop()
            interaction_phase = "waiting_for_input"
            yield session_log, "Input cancelled.", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(interactive=True, value=original_input), gr.update(), gr.update()
            return

    interaction_phase = "generating_response"
    settings = get_model_settings(temporary.MODEL_NAME)

    search_results = None
    if web_search_enabled:
        yield session_log, "🔍 Performing web search...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        search_results = await asyncio.to_thread(utility.web_search, original_input)
        status = "✅ Web search completed." if search_results and not search_results.startswith("Error") else "⚠️ No web results."
        session_log[-1]['content'] = f"AI-Chat:\n{status}"
        yield session_log, status, update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()

    q = queue.Queue()
    cancel_event = threading.Event()
    visible_response = ""

    def run_generator():
        try:
            for chunk in get_response_stream(
                session_log, settings, web_search_enabled, search_results,
                cancel_event, llm_state, models_loaded_state
            ):
                chunk = re.sub(r'^AI-Chat:\s*', '', chunk, flags=re.IGNORECASE)
                q.put(chunk)
            q.put(None)
        except Exception as e:
            q.put(f"Error: {str(e)}")

    thread = threading.Thread(target=run_generator, daemon=True)
    thread.start()

    while True:
        try:
            chunk = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        if chunk is None:
            break

        if cancel_flag:
            cancel_event.set()
            session_log[-1]['content'] = "AI-Chat:\nGeneration cancelled."
            yield session_log, "Cancelled", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
            break

        if chunk == "<CANCELLED>":
            session_log[-1]['content'] = "AI-Chat:\nGeneration cancelled."
            yield session_log, "Cancelled", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
            break

        if isinstance(chunk, str) and chunk.startswith("Error:"):
            print(f"Interface error: {chunk}"); time.sleep(3)
            session_log[-1]['content'] = f"AI-Chat:\n{chunk}"
            yield session_log, f"⚠️ {chunk}", update_action_button("waiting_for_input"), False, loaded_files, "waiting_for_input", gr.update(interactive=True), gr.update(), gr.update()
            return

        visible_response += chunk
        session_log[-1]['content'] = f"{visible_response.strip()}"
        yield session_log, "Streaming Response...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        await asyncio.sleep(0.05)

    if visible_response:
        links_match = re.search(r'\nLinks:\n(.*?)$', visible_response, re.DOTALL)
        clean_response = re.sub(r'\nLinks:\n.*?$', '', visible_response, flags=re.DOTALL).strip()
        response_lines = len([line for line in clean_response.split('\n') if line.strip()])
        
        if summary_enabled and response_lines > 4:
            temporary.CURRENT_SUMMARY = generate_summary(clean_response, llm_state)
            final_content = f"{clean_response}\n\nSUMMARY:\n{temporary.CURRENT_SUMMARY}"
        else:
            temporary.CURRENT_SUMMARY = ""
            final_content = clean_response
            
        if links_match:
            final_content += f"\n\nLinks:\n{links_match.group(1).strip()}"

        final_content = f"AI-Chat:\n{final_content.strip()}"
        session_log[-1]['content'] = final_content
        utility.save_session_history(session_log, temporary.session_attached_files)

    interaction_phase = "waiting_for_input"
    yield session_log, "✅ Response ready", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(interactive=True, value=""), gr.update(), gr.update()

def launch_interface():
    global demo
    with gr.Blocks(
        title="Conversation-Gradio-Gguf",
        css="""
        .scrollable { overflow-y: auto }
        .half-width { width: 80px !important }
        .double-height { height: 80px !important }
        .clean-elements { gap: 4px !important; margin-bottom: 4px !important }
        .send-button-green { background-color: green !important; color: white !important }
        .send-button-orange { background-color: orange !important; color: white !important }
        .send-button-red { background-color: red !important; color: white !important }
        """
    ) as demo:
        model_folder_state = gr.State(temporary.MODEL_FOLDER)
        states = {
            "attached_files": gr.State([]),
            "models_loaded": gr.State(False),
            "llm": gr.State(None),
            "cancel_flag": gr.State(False),
            "interaction_phase": gr.State("waiting_for_input"),
            "is_reasoning_model": gr.State(False),
            "selected_panel": gr.State("History"),
            "expanded_state": gr.State(True),
            "model_settings": gr.State({}),
            "web_search_enabled": gr.State(False),
            "summary_enabled": gr.State(False)
        }
        conversation_components = {}

        with gr.Tabs():
            with gr.Tab("Interaction"):
                with gr.Row():
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as left_column_expanded:
                        toggle_button_expanded = gr.Button("Chat-Linux-Gguf", variant="secondary")
                        panel_toggle = gr.Radio(choices=["History", "Attach"], label="Panel Mode", value="History")
                        with gr.Group(visible=False) as attach_group:
                            attach_files = gr.UploadButton(
                                "Add Attach Files",
                                file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS],
                                file_count="multiple",
                                variant="secondary"
                            )
                            attach_slots = [gr.Button("Attach Slot Free", variant="primary", visible=False) for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)]
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session...", variant="secondary")
                            buttons = {"session": [gr.Button(f"History Slot {i+1}", variant="primary", visible=False) for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)]}

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_collapsed = gr.Button("CGG", variant="secondary")
                        new_session_btn_collapsed = gr.Button("New", variant="secondary")
                        add_attach_files_collapsed = gr.UploadButton(
                            "Add",
                            file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS],
                            file_count="multiple",
                            variant="secondary"
                        )

                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        conversation_components["session_log"] = gr.Chatbot(
                            label="Session Log",
                            height=temporary.SESSION_LOG_HEIGHT,
                            elem_classes=["scrollable"],
                            type="messages"
                        )
                        with gr.Row(elem_classes=["clean_elements"]):
                            action_buttons = {
                                "web_search": gr.Button("🌐 Web-Search", variant="secondary", scale=1),
                                "summary": gr.Button("📝 Summary", variant="secondary", scale=1)
                            }
                        initial_max_lines = max(3, int(((temporary.SESSION_LOG_HEIGHT - 100) / 10) / 2.5) - 6)
                        temporary.USER_INPUT_MAX_LINES = initial_max_lines
                        conversation_components["user_input"] = gr.Textbox(
                            label="User Input",
                            lines=3,
                            max_lines=initial_max_lines,
                            interactive=False,
                            placeholder="Enter text here..."
                        )
                        conversation_components["user_input"].change(
                            fn=update_session_log_height,
                            inputs=[conversation_components["user_input"]],
                            outputs=[conversation_components["session_log"]]
                        )
                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["action"] = gr.Button("Send Input", variant="secondary", elem_classes=["send-button-green"], scale=10)
                            action_buttons["edit_previous"] = gr.Button("Edit Previous", variant="primary", scale=1)
                            action_buttons["copy_response"] = gr.Button("Copy Output", variant="primary", scale=1)

                with gr.Row():
                    status_text = gr.Textbox(label="Status", interactive=False, value="Select model on Configuration page.", scale=30)
                    exit_button = gr.Button("Exit", variant="stop", elem_classes=["double-height"], min_width=110)
                    exit_button.click(fn=shutdown_program, inputs=[states["llm"], states["models_loaded"]])

            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    config_components = {}
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Hardware Info...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            selected_gpu=gr.Textbox(
                                label="Selected GPU",
                                value=str(temporary.SELECTED_GPU),
                                interactive=False,
                                scale=2
                            ),
                            cuda_version=gr.Textbox(
                                label="CUDA Version",
                                value=temporary.CUDA_VERSION,
                                interactive=False,
                                scale=2
                            ),
                            compute_capability=gr.Textbox(
                                label="Compute Capability",
                                value=temporary.COMPUTE_CAPABILITY,
                                interactive=False,
                                scale=2
                            ),
                            vram_mb=gr.Textbox(
                                label="VRAM (MB)",
                                value=str(temporary.VRAM_MB),
                                interactive=False,
                                scale=2
                            ),
                            system_ram_mb=gr.Textbox(
                                label="System RAM (MB)",
                                value=str(temporary.SYSTEM_RAM_MB),
                                interactive=False,
                                scale=2
                            ),
                            ddr_level=gr.Textbox(
                                label="DDR Level",
                                value=temporary.DDR_LEVEL,
                                interactive=False,
                                scale=2
                            )
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Model Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        model_path_display = gr.Textbox(
                            label="Model Folder",
                            value=temporary.MODEL_FOLDER,
                            interactive=False,
                            scale=10
                        )
                        available_models = temporary.AVAILABLE_MODELS or get_available_models()
                        base_choices = ["Select_a_model..."]
                        if available_models != ["Select_a_model..."]:
                            available_models = base_choices + [m for m in available_models if m not in base_choices]
                        else:
                            available_models = base_choices
                        default_model = (
                            temporary.MODEL_NAME
                            if temporary.MODEL_NAME in available_models and temporary.MODEL_NAME not in base_choices
                            else (available_models[1] if len(available_models) > 1 else "Select_a_model...")
                        )
                        config_components["model"] = gr.Dropdown(
                            choices=available_models,
                            label="Select Model File",
                            value=default_model,
                            allow_custom_value=False,
                            scale=10
                        )
                        keywords_display = gr.Textbox(
                            label="Keywords Detected",
                            interactive=False,
                            value="",
                            scale=10
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            ctx=gr.Dropdown(
                                choices=temporary.CTX_OPTIONS,
                                label="Context Size",
                                value=temporary.CONTEXT_SIZE,
                                scale=5
                            ),
                            batch=gr.Dropdown(
                                choices=temporary.BATCH_OPTIONS,
                                label="Batch Size",
                                value=temporary.BATCH_SIZE,
                                scale=5
                            ),
                            temp=gr.Dropdown(
                                choices=temporary.TEMP_OPTIONS,
                                label="Temperature",
                                value=temporary.TEMPERATURE,
                                scale=5
                            ),
                            repeat=gr.Dropdown(
                                choices=temporary.REPEAT_OPTIONS,
                                label="Repeat Penalty",
                                value=temporary.REPEAT_PENALTY,
                                scale=5
                            )
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        browse_button = gr.Button("Browse Folder", variant="secondary")
                        config_components.update(
                            load_models=gr.Button("Load Model", variant="secondary"),
                            unload=gr.Button("Unload Model", variant="primary")
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Program Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        custom_components = {
                            "max_history_slots": gr.Dropdown(
                                choices=temporary.HISTORY_SLOT_OPTIONS,
                                label="Max History Slots",
                                value=temporary.MAX_HISTORY_SLOTS,
                                scale=5
                            ),
                            "session_log_height": gr.Dropdown(
                                choices=temporary.SESSION_LOG_HEIGHT_OPTIONS,
                                label="Session Log Height",
                                value=temporary.SESSION_LOG_HEIGHT,
                                scale=5
                            ),
                            "max_attach_slots": gr.Dropdown(
                                choices=temporary.ATTACH_SLOT_OPTIONS,
                                label="Max Attach Slots",
                                value=temporary.MAX_ATTACH_SLOTS,
                                scale=5
                            )
                        }
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Critical Actions...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components["save_settings"] = gr.Button("Save Settings", variant="primary")
                        custom_components["delete_all_history"] = gr.Button("Delete All History", variant="stop")
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("About Program...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("[Chat-Linux-Gguf](https://github.com/wiseman-timelord/Chat-Linux-Gguf) by [Wiseman-Timelord](https://github.com/wiseman-timelord).")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            status_settings=gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Select model on Configuration page. Check terminal for issues.",
                                scale=20
                            ),
                            shutdown=gr.Button(
                                "Exit",
                                variant="stop",
                                elem_classes=["double-height"],
                                min_width=110
                            ).click(
                                fn=shutdown_program,
                                inputs=[states["llm"], states["models_loaded"]]
                            )
                        )

        def handle_edit_previous(session_log):
            if len(session_log) < 2:
                return session_log, gr.update(), "No previous input to edit."
            new_log = session_log[:-2]
            last_user_input = session_log[-2]['content'].replace("User:\n", "", 1)
            return new_log, gr.update(value=last_user_input), "Previous input restored."

        model_folder_state.change(
            fn=lambda f: setattr(temporary, "MODEL_FOLDER", f) or None,
            inputs=[model_folder_state],
            outputs=[]
        ).then(fn=update_model_list, inputs=[model_folder_state], outputs=[config_components["model"]]).then(
            fn=lambda f: f"Model directory updated to: {f}", inputs=[model_folder_state], outputs=[status_text]
        )

        start_new_session_btn.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[conversation_components["session_log"], status_text, conversation_components["user_input"], states["web_search_enabled"]]
        ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"]).then(
            fn=lambda: [], inputs=[], outputs=[states["attached_files"]]
        )

        new_session_btn_collapsed.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[conversation_components["session_log"], status_text, conversation_components["user_input"], states["web_search_enabled"]]
        ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"]).then(
            fn=lambda: [], inputs=[], outputs=[states["attached_files"]]
        )

        attach_files.upload(
            fn=process_attach_files,
            inputs=[attach_files, states["attached_files"], states["models_loaded"]],
            outputs=[status_text, states["attached_files"]]
        ).then(fn=lambda files: update_file_slot_ui(files, True), inputs=[states["attached_files"]], outputs=attach_slots + [attach_files])

        add_attach_files_collapsed.upload(
            fn=process_attach_files,
            inputs=[add_attach_files_collapsed, states["attached_files"], states["models_loaded"]],
            outputs=[status_text, states["attached_files"]]
        ).then(fn=lambda files: update_file_slot_ui(files, True), inputs=[states["attached_files"]], outputs=attach_slots + [add_attach_files_collapsed])

        browse_button.click(
            fn=browse_on_click,
            inputs=[model_folder_state],
            outputs=[model_folder_state]
        ).then(fn=update_model_list, inputs=[model_folder_state], outputs=[config_components["model"]]).then(
            fn=lambda f: f, inputs=[model_folder_state], outputs=[model_path_display]
        ).then(fn=lambda f: f"Model directory updated to: {f}", inputs=[model_folder_state], outputs=[status_text])

        action_buttons["action"].click(
            fn=lambda phase: True if phase == "generating_response" else False,
            inputs=[states["interaction_phase"]],
            outputs=[states["cancel_flag"]]
        ).then(
            fn=conversation_interface,
            inputs=[
                conversation_components["user_input"], conversation_components["session_log"], states["attached_files"],
                states["is_reasoning_model"], states["cancel_flag"], states["web_search_enabled"], states["interaction_phase"],
                states["llm"], states["models_loaded"], states["summary_enabled"]
            ],
            outputs=[
                conversation_components["session_log"], status_text, action_buttons["action"], states["cancel_flag"],
                states["attached_files"], states["interaction_phase"], conversation_components["user_input"],
                states["web_search_enabled"], states["summary_enabled"]
            ]
        ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"])

        action_buttons["copy_response"].click(fn=copy_last_response, inputs=[conversation_components["session_log"]], outputs=[status_text])

        action_buttons["web_search"].click(
            fn=lambda enabled: not enabled,
            inputs=[states["web_search_enabled"]],
            outputs=[states["web_search_enabled"]]
        ).then(
            lambda state: gr.update(variant="primary" if state else "secondary"),
            inputs=[states["web_search_enabled"]], outputs=[action_buttons["web_search"]]
        )

        action_buttons["summary"].click(
            fn=lambda enabled: not enabled,
            inputs=[states["summary_enabled"]],
            outputs=[states["summary_enabled"]]
        ).then(
            lambda state: gr.update(variant="primary" if state else "secondary"),
            inputs=[states["summary_enabled"]], outputs=[action_buttons["summary"]]
        )

        action_buttons["edit_previous"].click(
            fn=handle_edit_previous,
            inputs=[conversation_components["session_log"]],
            outputs=[conversation_components["session_log"], conversation_components["user_input"], status_text]
        )

        for i, btn in enumerate(attach_slots):
            btn.click(
                fn=lambda files, idx=i: utility.eject_file(files, idx, True),
                inputs=[states["attached_files"]],
                outputs=[states["attached_files"], status_text] + attach_slots + [attach_files]
            )

        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=load_session_by_index,
                inputs=[gr.State(value=i)],
                outputs=[conversation_components["session_log"], states["attached_files"], status_text]
            ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"]).then(
                fn=lambda files: update_file_slot_ui(files, True), inputs=[states["attached_files"]], outputs=attach_slots + [attach_files]
            )

        panel_toggle.change(fn=lambda panel: panel, inputs=[panel_toggle], outputs=[states["selected_panel"]])

        config_components["model"].change(
            fn=handle_model_selection,
            inputs=[config_components["model"], model_folder_state],
            outputs=[model_folder_state, config_components["model"], status_text]
        ).then(
            fn=lambda model_name: get_model_settings(model_name)["is_reasoning"],
            inputs=[config_components["model"]],
            outputs=[states["is_reasoning_model"]]
        ).then(
            fn=lambda model_name: get_model_settings(model_name),
            inputs=[config_components["model"]],
            outputs=[states["model_settings"]]
        ).then(
            fn=update_panel_choices,
            inputs=[states["model_settings"], states["selected_panel"]],
            outputs=[panel_toggle, states["selected_panel"]]
        ).then(
            fn=lambda model_settings: "none" if not model_settings.get("detected_keywords", []) else ", ".join(model_settings.get("detected_keywords", [])),
            inputs=[states["model_settings"]],
            outputs=[keywords_display]
        )

        states["selected_panel"].change(
            fn=lambda panel: (gr.update(visible=panel == "Attach"), gr.update(visible=panel == "History")),
            inputs=[states["selected_panel"]],
            outputs=[attach_group, history_slots_group]
        )

        for comp in [config_components[k] for k in ["ctx", "batch", "temp", "repeat", "model"]]:
            comp.change(
                fn=update_config_settings,
                inputs=[config_components[k] for k in ["ctx", "batch", "temp", "repeat", "model"]],
                outputs=[status_text]
            )

        config_components["unload"].click(
            fn=unload_models,
            inputs=[states["llm"], states["models_loaded"]],
            outputs=[status_text, states["llm"], states["models_loaded"]]
        ).then(fn=lambda: gr.update(interactive=False), outputs=[conversation_components["user_input"]])

        config_components["load_models"].click(
            fn=set_loading_status,
            outputs=[status_text]
        ).then(
            fn=load_models,
            inputs=[model_folder_state, config_components["model"], states["llm"], states["models_loaded"]],
            outputs=[status_text, states["models_loaded"], states["llm"], states["models_loaded"]]
        ).then(
            fn=lambda status, ml: (status, gr.update(interactive=ml)),
            inputs=[status_text, states["models_loaded"]],
            outputs=[status_text, conversation_components["user_input"]]
        )

        config_components["save_settings"].click(fn=save_all_settings, outputs=[status_text])

        custom_components["delete_all_history"].click(
            fn=utility.delete_all_session_histories,
            outputs=[status_text]
        ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"])

        custom_components["session_log_height"].change(
            fn=set_session_log_base_height,
            inputs=[custom_components["session_log_height"]],
            outputs=[conversation_components["session_log"]]
        )

        custom_components["max_history_slots"].change(
            fn=lambda s: setattr(temporary, "MAX_HISTORY_SLOTS", int(s)) or None,
            inputs=[custom_components["max_history_slots"]],
            outputs=[]
        ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"])

        custom_components["max_attach_slots"].change(
            fn=lambda s: setattr(temporary, "MAX_ATTACH_SLOTS", int(s)) or None,
            inputs=[custom_components["max_attach_slots"]],
            outputs=[]
        ).then(fn=lambda files: update_file_slot_ui(files, True), inputs=[states["attached_files"]], outputs=attach_slots + [attach_files])

        toggle_button_expanded.click(fn=lambda state: not state, inputs=[states["expanded_state"]], outputs=[states["expanded_state"]])
        toggle_button_collapsed.click(fn=lambda state: not state, inputs=[states["expanded_state"]], outputs=[states["expanded_state"]])
        states["expanded_state"].change(
            fn=lambda state: [gr.update(visible=state), gr.update(visible=not state)],
            inputs=[states["expanded_state"]],
            outputs=[left_column_expanded, left_column_collapsed]
        )

        demo.load(
            fn=get_initial_model_value,
            inputs=[],
            outputs=[config_components["model"], states["is_reasoning_model"]]
        ).then(
            fn=lambda model_name: get_model_settings(model_name),
            inputs=[config_components["model"]],
            outputs=[states["model_settings"]]
        ).then(
            fn=update_panel_choices,
            inputs=[states["model_settings"], states["selected_panel"]],
            outputs=[panel_toggle, states["selected_panel"]]
        ).then(fn=update_session_buttons, inputs=[], outputs=buttons["session"]).then(
            fn=lambda files: update_file_slot_ui(files, True), inputs=[states["attached_files"]], outputs=attach_slots + [attach_files]
        ).then(
            fn=lambda model_settings: "none" if not model_settings.get("detected_keywords", []) else ", ".join(model_settings.get("detected_keywords", [])),
            inputs=[states["model_settings"]],
            outputs=[keywords_display]
        )

        status_text.change(fn=lambda status: status, inputs=[status_text], outputs=[config_components["status_settings"]])

    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, show_api=False)

if __name__ == "__main__":
    launch_interface()
