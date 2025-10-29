import os
import io
import json
import threading
import time
import random
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np

# Import processing helpers from face1 (we'll reuse some functions)
from face1 import overlay_transparent, load_overlays, face_cascade

# LLM imports
import requests

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not available. llama.cpp features will be disabled.")

# Check if Ollama is available
try:
    import requests
    OLLAMA_AVAILABLE = True
    print("Ollama support: Available (using requests)")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: Ollama not available. Install ollama package.")

# Story templates for Halloween stories
STORY_TEMPLATES = [
    # Spooky
    "{name}, the office lights flicker as your {costume} passes—spirits linger by the copier, whispering about your missing {treat}. Beware the next “reply all”; it won’t come from the living.",
    "The full moon shines on your desk, {name}, and the scent of {treat} lures shadows from the breakroom. Before dawn, you’ll discover who’s been haunting the printer—look closely at the reflection in the monitor.",
    "Your inbox glows with eerie light, {name}, each unread message pulsing like a heartbeat. The ghosts of deadlines past demand tribute—perhaps a piece of {treat} will appease them.",
    "As you sip stale coffee under fluorescent moons, {name}, your {costume} begins to whisper. The whispers spell your name… backward.",
    "The conference room is locked, {name}, yet laughter echoes inside. The spirits of meetings-that-should-have-been-emails hunger for your {treat} tonight.",
    # Funny
    "{name}, the spirits admire your {costume}—they say you’re overdressed for casual Friday but underdressed for the apocalypse. Guard your {treat}, or HR may classify it as a taxable perk.",
    "I see you, {name}, triumphantly wielding {treat} like office contraband. Beware: by 3 PM, your snack will vanish faster than productivity after a team-building exercise.",
    "Your future holds caffeine and confusion, {name}. The {treat} will sustain you through endless meetings, but beware—the ghost of Outlook Crashes Past still roams.",
    "Dressed as {costume}, {name}, you’ll win the costume contest… but only because everyone else forgot to unmute. Celebrate with {treat}, before Finance audits your joy.",
    "The stars foresee {treat} crumbs on your keyboard, {name}. Fear not—the IT spirits thrive on chaos and caramel.",
    # Adventurous
    "{name}, destiny calls from the 13th floor where no badge has worked since 2008. Only your {costume} and a handful of {treat} can guide you safely back before the next status update.",
    "The office air vents whisper maps to hidden breakroom realms, {name}. Follow the trail of {treat} wrappers—what you find will change your quarterly goals forever.",
    "With your {costume} as armor and {treat} as offering, {name}, you’ll brave the haunted storage room. There, the ancient stapler awaits its chosen wielder.",
    "The spirits of innovation summon you, {name}. They promise glory—and unlimited {treat}—if you survive the labyrinth of meetings unscathed.",
    "Tonight, {name}, you’ll chase the phantom of lost Wi-Fi through the cubicle maze. Only {treat} and courage will see you through.",
    # Halloweeny
    "Pumpkins grin from every cubicle, {name}, as your {costume} glows brighter than the copier light. Share your {treat}, and the office spirits might spare your lunch from vanishing.",
    "Candles flicker beside your monitor, {name}, and bats circle the breakroom fridge. The prophecy says whoever offers {treat} will summon the Great Pumpkin of Payroll.",
    "{name}, the aroma of {treat} drifts through the hall like a spell. Soon, coworkers will gather like zombies for the annual potluck of destiny.",
    "Your {costume} dazzles even the undead in Accounting, {name}. Before the night ends, one will reveal the sacred secret: where the leftover candy is hidden.",
    "As the moon rises over the office park, {name}, your {costume} marks you chosen for the Candy Coven. Guard your {treat}—and your PTO balance—with care."
]

MAIN_PROMPT_TEMPLATE = (
    "You are a mischievous Halloween fortune teller speaking from beyond the veil. "
    "Generate a short fortune for {name}, who is dressed as {costume}, and whose favorite "
    "treat is {treat}. The fortune should follow these rules:\n"
    "1. Incorporate the name, costume, and treat.\n"
    "2. Be spooky and mysterious.\n"
    "3. Be 3-4 sentences and 50-75 words long.\n"
    "4. Use a fortune teller's voice (e.g., 'I see...', 'Beware...', 'The spirits whisper...').\n"
    "5. Include Halloween imagery (moons, shadows, candles, spirits, etc.).\n"
    "6. End with a chilling or humorous twist.\n"
    "7. Take place in an office setting.\n\n"
    "Here are a few sample fortunes for inspiration:\n{examples}\n\n"
    "Now deliver the fortune in a single coherent block of text."
)

OLLAMA_DEFAULT_MODEL = "llama3.2:1b"

# LLM instance
llm_instance = None
llm_type = None  # 'llama_cpp' or 'ollama'
llm_model_name = None
prompt_config = None
debug_config = None

def debug_log(message, category="INFO"):
    """Log debug messages based on configuration."""
    if not debug_config or not debug_config.get('enabled', False):
        return
    
    log_level = debug_config.get('log_level', 'INFO')
    if category == "ERROR" and debug_config.get('show_errors', True):
        print(f"[DEBUG-ERROR] {message}")
    elif category == "INFO" and log_level in ["INFO", "DEBUG"]:
        print(f"[DEBUG-INFO] {message}")
    elif category == "DEBUG" and log_level == "DEBUG":
        print(f"[DEBUG-DEBUG] {message}")

def debug_log_prompt(prompt, llm_params=None):
    """Log prompt and LLM parameters if debug mode is enabled."""
    if not debug_config or not debug_config.get('enabled', False):
        return
    
    if debug_config.get('show_prompts', True):
        debug_log(f"Main template: {prompt}")
    
    if debug_config.get('show_llm_parameters', True) and llm_params:
        debug_log(f"LLM parameters: {llm_params}")

def debug_log_story_generation(name, costume, treat, story):
    """Log story generation details if debug mode is enabled."""
    if not debug_config or not debug_config.get('enabled', False):
        return
    
    if debug_config.get('show_story_generation', True):
        debug_log(f"Generating story for: {name} as {costume} with {treat}")
        debug_log(f"Generated story: {story}")

def debug_log_api_call(endpoint, method="GET", data=None):
    """Log API calls if debug mode is enabled."""
    if not debug_config or not debug_config.get('enabled', False):
        return
    
    if debug_config.get('show_api_calls', True):
        debug_log(f"API Call: {method} {endpoint}")
        if data:
            debug_log(f"API Data: {data}")

def load_prompt_config():
    """Load prompt configuration from file."""
    global prompt_config, debug_config
    config_path = os.path.join(os.path.dirname(__file__), "prompt_config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            prompt_config = json.load(f)

        # Remove prompt instructions that are now hard-coded in this module.
        for legacy_key in ("system_prompt", "main_prompt_template", "sample_templates"):
            prompt_config.pop(legacy_key, None)
        
        # Extract debug configuration
        debug_config = prompt_config.get('debug_mode', {})
        
        print("Prompt configuration loaded successfully.")
        if debug_config.get('enabled', False):
            print("Debug mode is ENABLED")
        return True
    except Exception as e:
        print(f"Error loading prompt config: {e}")
        # Use default configuration
        prompt_config = {
            "llm_parameters": {
                "max_tokens": 200,
                "temperature": 0.8,
                "top_p": 0.9,
                "stop": ["\n\n", "###", "---"]
            },
            "debug_mode": {
                "enabled": False,
                "log_level": "INFO",
                "show_prompts": True,
                "show_llm_parameters": True,
                "show_story_generation": True,
                "show_api_calls": True,
                "show_errors": True
            },
            "fallback_templates": [
                "{name} haunts the office in a {costume}, lurking near the {treat}. Employees scream!"
            ]
        }
        debug_config = prompt_config.get('debug_mode', {})
        return False

def save_prompt_config():
    """Save current prompt configuration to file."""
    if not prompt_config:
        return False
    config_path = os.path.join(os.path.dirname(__file__), "prompt_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_config, f, indent=2, ensure_ascii=False)
        print("Prompt configuration saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving prompt config: {e}")
        return False

def initialize_llm():
    """Initialize the local LLM model - try Ollama first, then llama.cpp."""
    global llm_instance, llm_type, llm_model_name
    
    # Try Ollama first (easier to use and more reliable)
    if OLLAMA_AVAILABLE:
        try:
            print("Attempting to connect to Ollama...")
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✓ Connected to Ollama!")
                llm_type = "ollama"
                llm_model_name = OLLAMA_DEFAULT_MODEL
                llm_instance = None  # We'll use requests directly for Ollama
                return True
        except Exception as e:
            print(f"Ollama not available: {e}")
            print("Falling back to llama.cpp...")
    
    # Try llama.cpp
    if LLAMA_CPP_AVAILABLE:
        try:
            model_paths = [
                os.path.join(os.path.dirname(__file__), "models", "llama-2-7b-chat.Q2_K.gguf"),
                os.path.join(os.path.dirname(__file__), "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Found model: {path}")
                    break
            
            if model_path:
                print(f"Loading LLM model from: {model_path}")
                llm_instance = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                llm_type = "llama_cpp"
                llm_model_name = os.path.basename(model_path)
                print("✓ LLM initialized successfully (llama.cpp)!")
                return True
        except Exception as e:
            print(f"ERROR initializing llama.cpp: {e}")
            import traceback
            traceback.print_exc()
    
    print("✗ No LLM backend available. Install ollama or llama-cpp-python.")
    return False

def generate_story_with_llm(name, costume_display, treat_display):
    """Generate a Halloween story using the local LLM."""
    debug_log(f"Generating story for: {name} as {costume_display} with {treat_display}", "DEBUG")
    
    if not llm_type:
        debug_log("LLM not initialized, using fallback story generation.", "ERROR")
        print("ERROR: LLM not initialized - check if LLM initialized successfully at startup")
        return generate_story_fallback(name, costume_display, treat_display)
    
    if not prompt_config:
        debug_log("Prompt config not loaded, using fallback story generation.", "ERROR")
        print("ERROR: Prompt config is None")
        return generate_story_fallback(name, costume_display, treat_display)

    model_name = llm_model_name or 'unknown'
    debug_log(f"Using LLM backend '{llm_type}' with model '{model_name}'", "DEBUG")

    # Format the examples
    examples_filled = [
        tpl.format(
            name=name or 'Someone',
            costume=costume_display or 'mysterious costume',
            treat=treat_display or 'mysterious treat'
        )
        for tpl in STORY_TEMPLATES
    ]
    examples_text = '\n'.join(examples_filled)
    
    # Create the full prompt
    prompt = MAIN_PROMPT_TEMPLATE.format(
        name=name or 'Someone',
        costume=costume_display or 'mysterious costume',
        treat=treat_display or 'mysterious treat',
        examples=examples_text
    )

    # Get LLM parameters from config
    llm_params = prompt_config.get('llm_parameters', {})
    
    # Debug logging
    debug_log_prompt(prompt, llm_params)
    
    try:
        debug_log("Calling LLM to generate story...", "DEBUG")
        print(f"[DEBUG] Using LLM backend: {llm_type} | model: {model_name}")
        print(f"[DEBUG] Calling LLM ({llm_type}) with prompt length: {len(prompt)} chars")
        
        # Call the appropriate LLM backend
        if llm_type == "ollama":
            # Use Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": OLLAMA_DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": llm_params.get('max_tokens', 200),
                        "temperature": llm_params.get('temperature', 0.8),
                        "top_p": llm_params.get('top_p', 0.9),
                        "repeat_penalty": llm_params.get('repeat_penalty', 1.18),
                        "stop": llm_params.get('stop', ["\n\n", "###", "---"])
                    }
                }
            )
            response_json = response.json()
            story = response_json.get('response', '').strip()
            print(f"[DEBUG] Ollama response: {story[:200] if story else 'EMPTY'}")
            
        elif llm_type == "llama_cpp":
            # Use llama.cpp
            try:
                response = llm_instance(
                    prompt,
                    max_tokens=llm_params.get('max_tokens', 200),
                    temperature=llm_params.get('temperature', 0.8),
                    top_p=llm_params.get('top_p', 0.9),
                    stop=llm_params.get('stop', ["\n\n", "###", "---"]),
                    echo=False
                )
            except Exception as api_error:
                print(f"[ERROR] First API call failed: {api_error}")
                print("[DEBUG] Trying alternative API format...")
                response = llm_instance(
                    prompt,
                    max_tokens=llm_params.get('max_tokens', 200),
                    temperature=llm_params.get('temperature', 0.8),
                    top_p=llm_params.get('top_p', 0.9),
                    stop=llm_params.get('stop', ["\n\n", "###", "---"])
                )
        
        # Parse response based on LLM type
        if llm_type == "ollama":
            # story already extracted above
            pass
        elif llm_type == "llama_cpp":
            # Parse llama.cpp response
            story = None
            if isinstance(response, dict):
                print(f"[DEBUG] LLM response keys: {list(response.keys())}")
                
                # Try 'choices' format (OpenAI-compatible)
                if 'choices' in response and len(response['choices']) > 0:
                    choice = response['choices'][0]
                    if isinstance(choice, dict):
                        story = choice.get('text', '').strip()
                        print(f"[DEBUG] Extracted text from choices: {story[:100] if story else 'EMPTY'}")
                
                # Try 'text' key
                if not story and 'text' in response:
                    story = response['text'].strip()
            elif isinstance(response, str):
                story = response.strip()
                
            if not story:
                print(f"[ERROR] Could not extract story from response: {response}")
                story = ""
        
        print(f"[DEBUG] Extracted story length: {len(story) if story else 0} chars")
        print(f"[DEBUG] Extracted story preview: {story[:200] if story else 'EMPTY'}")
        
        if story and len(story) > 0:
            debug_log_story_generation(name, costume_display, treat_display, story)
            debug_log("Story generated successfully using LLM", "INFO")
            return story
        else:
            print("[ERROR] LLM returned empty story!")
            debug_log("LLM returned empty story", "ERROR")
            print(f"[ERROR] Full response was: {response}")
    
    except Exception as e:
        print(f"[ERROR] Exception during LLM call: {e}")
        debug_log(f"Error generating story with LLM: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    
    # Fallback to template-based generation
    debug_log("Using fallback story generation", "INFO")
    return generate_story_fallback(name, costume_display, treat_display)

def generate_story_fallback(name, costume_display, treat_display):
    """Fallback story generation using templates."""
    # Use configurable fallback templates if available
    if prompt_config and 'fallback_templates' in prompt_config:
        templates = prompt_config['fallback_templates']
    else:
        templates = STORY_TEMPLATES
    
    template = random.choice(templates)
    return template.format(
        name=name or "Someone", 
        costume=costume_display or "mysterious costume", 
        treat=treat_display or "mysterious treat"
    )

def generate_story(name, costume, treat):
    """Generate a Halloween story using the provided name, costume, and treat."""
    # Map costume filenames to display names
    costume_display_map = {
        "fangs.png": "Fangs",
        "horns.png": "Horns",
        "yd1.png": "Yellow Devil 1",
        "yd2.png": "Yellow Devil 2",
        "yd3.png": "Yellow Devil 3",
        "yd4.png": "Yellow Devil 4",
        "yd5.png": "Yellow Devil 5",
        "yp1.png": "Yellow Pumpkin 1",
        "yp2.png": "Yellow Pumpkin 2",
        "yp3.png": "Yellow Pumpkin 3",
        "yp4.png": "Yellow Pumpkin 4",
        "yp5.png": "Yellow Pumpkin 5",
        "yw1.png": "Yellow Witch 1",
        "yw2.png": "Yellow Witch 2",
        "yw3.png": "Yellow Witch 3",
        "yw4.png": "Yellow Witch 4",
        "yw5.png": "Yellow Witch 5",
        "witch": "Witch",
        "pumpkin": "Pumpkin",
        "devil": "Devil"
    }

    # Map treat filenames to display names
    treat_display_map = {
        "xca1.png": "Spooky Candy A",
        "xca2.png": "Spooky Candy B",
        "xca3.png": "Spooky Candy C",
        "xca4.png": "Spooky Candy D",
        "xch1.png": "Chocolate Treat 1",
        "xch2.png": "Chocolate Treat 2",
    "xch3.png": "Chocolate Treat 3",
    "xci1.png": "Chips 1",
    "xci2.png": "Chips 2",
    "xci3.png": "Chips 3",
        "candy": "Candy",
        "chocolate": "Chocolate",
        "chips": "Chips"
    }

    costume_key = (costume or '').strip().lower()
    treat_key = (treat or '').strip().lower()

    mapped_costume = costume_display_map.get(costume_key)
    if not mapped_costume:
        if costume_key.startswith('yw'):
            mapped_costume = 'Yellow Witch'
        elif costume_key.startswith('yd'):
            mapped_costume = 'Yellow Devil'
        elif costume_key.startswith('yp'):
            mapped_costume = 'Yellow Pumpkin'

    mapped_treat = treat_display_map.get(treat_key)
    if not mapped_treat:
        if treat_key.startswith('xca'):
            mapped_treat = 'Candy'
        elif treat_key.startswith('xch'):
            mapped_treat = 'Chocolate'
    elif treat_key.startswith('xci'):
            mapped_treat = 'Chips'

    costume_display = mapped_costume or (costume if costume else "mysterious costume")
    treat_display = mapped_treat or (treat if treat else "mysterious treat")
    
    return generate_story_with_llm(name, costume_display, treat_display)

app = Flask(__name__)

# Global state shared between HTTP handlers and capture thread
state = {
    "overlays_dir": os.path.join(os.path.dirname(__file__), "overlays"),
    "snapshots_dir": os.path.join(os.path.dirname(__file__), "snapshots"),
    "overlays": [],
    "current_idx": 0,
    "_overlays_signature": (),
    "overlay_lookup": {},
    "active_overlays": [],
    "show_face_box": False,
    "face_box_thickness": 1,
    "frame": None,
    "running": True,
    "nudge_mode": False,
    "nudge_overlay": None,
    "nudge_message": "",
    "fps": 0.0,
    "mode": "prod",
    "prod_frozen": False,
    "prod_frame": None,
    "prod_waiting": False,
    "last_snapshot_path": None,
    "story_text": "",
    "story_updated": None,
    "user_name": "",
    "user_costume": "",
    "user_treat": "",
}


def ensure_snapshot_dir():
    directory = state.get("snapshots_dir")
    if directory:
        os.makedirs(directory, exist_ok=True)


def compute_overlays_signature(directory):
    if not directory or not os.path.isdir(directory):
        return ()
    try:
        entries = os.listdir(directory)
    except OSError:
        return ()
    signature = []
    for name in sorted(entries):
        if not name.lower().endswith('.png') and name.lower() != 'config.json':
            continue
        path = os.path.join(directory, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        signature.append((name, int(stat.st_mtime), int(stat.st_size)))
    return tuple(signature)


def next_snapshot_path():
    directory = state.get("snapshots_dir")
    if not directory:
        return None
    ensure_snapshot_dir()
    existing_numbers = []
    for name in os.listdir(directory):
        if not name.startswith("snap") or not name.lower().endswith(".jpg"):
            continue
        suffix = name[4:-4]
        if suffix.isdigit():
            existing_numbers.append(int(suffix))
    next_idx = max(existing_numbers) + 1 if existing_numbers else 0
    while True:
        candidate = os.path.join(directory, f"snap{next_idx:02d}.jpg")
        if not os.path.exists(candidate):
            return candidate
        next_idx += 1


def save_snapshot(frame):
    path = next_snapshot_path()
    if path is None:
        return
    try:
        cv2.imwrite(path, frame)
        state["last_snapshot_path"] = os.path.basename(path)
    except Exception:
        # Swallow errors but clear last_snapshot_path so status does not lie
        state["last_snapshot_path"] = None


def list_overlay_names():
    return [ov.get("name") for ov in state.get("overlays", [])]


def reload_overlays(force=False):
    directory = state.get("overlays_dir")
    signature = compute_overlays_signature(directory)
    if not force and signature == state.get("_overlays_signature"):
        return

    overlays = load_overlays(directory) if directory and os.path.isdir(directory) else []
    state["overlays"] = overlays
    state["_overlays_signature"] = signature
    refresh_overlay_lookup()
    active = state.get("active_overlays", []) or []
    state["active_overlays"] = [name for name in active if get_overlay_entry(name)]
    if not overlays:
        state["current_idx"] = 0
    else:
        state["current_idx"] = state.get("current_idx", 0) % len(overlays)
    target = state.get("nudge_overlay")
    if target and not get_overlay_entry(target):
        state["nudge_overlay"] = None
        set_nudge_message("Nudge target unavailable; select another overlay.")


def persist_overlays_config():
    directory = state.get("overlays_dir")
    if not directory:
        return False
    cfg_path = os.path.join(directory, "config.json")
    data = {}
    for ov in state.get("overlays") or []:
        name = ov.get("name")
        cfg = ov.get("config", {}) or {}
        if not name:
            continue
        data[name] = {
            "scale": float(cfg.get("scale", 0.9)),
            "y_offset": float(cfg.get("y_offset", 0.55)),
            "x_anchor": float(cfg.get("x_anchor", 0.5)),
            "offset_x": round(float(cfg.get("offset_x", 0.0)), 3),
            "offset_y": round(float(cfg.get("offset_y", 0.0)), 3),
        }
        if "description" in cfg:
            data[name]["description"] = cfg["description"]
    tmp_path = cfg_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, cfg_path)
    state["_overlays_signature"] = compute_overlays_signature(directory)
    refresh_overlay_lookup()
    return True


def refresh_overlay_lookup():
    overlays = state.get("overlays") or []
    state["overlay_lookup"] = {
        ov.get("name", "").lower(): ov for ov in overlays if ov.get("name")
    }


def get_overlay_entry(name):
    if not name:
        return None
    lookup = state.get("overlay_lookup") or {}
    return lookup.get(name.lower())


def get_overlay_config_snapshot(name):
    entry = get_overlay_entry(name)
    if entry is None:
        return None
    cfg = entry.get("config", {}) or {}
    return {
        "scale": float(cfg.get("scale", 0.9)),
        "y_offset": float(cfg.get("y_offset", 0.55)),
        "x_anchor": float(cfg.get("x_anchor", 0.5)),
        "offset_x": float(cfg.get("offset_x", 0.0)),
        "offset_y": float(cfg.get("offset_y", 0.0)),
    }


def set_nudge_message(text):
    state["nudge_message"] = text or ""


def ensure_nudge_target():
    target = state.get("nudge_overlay")
    entry = get_overlay_entry(target)
    if entry is not None:
        return target
    active = state.get("active_overlays") or []
    for name in reversed(active):
        if get_overlay_entry(name):
            state["nudge_overlay"] = name
            return name
    state["nudge_overlay"] = None
    return None


def add_overlay_by_name(name):
    reload_overlays()
    overlays = state.get("overlays") or []
    if not overlays:
        return None
    normalized = (name or "").lower().strip()
    if not normalized.endswith('.png'):
        normalized = f"{normalized}.png"
    entry = get_overlay_entry(normalized)
    if entry is None:
        return None
    actual_name = entry.get("name")
    active = state.setdefault("active_overlays", [])
    if actual_name in active:
        active.remove(actual_name)
    active.append(actual_name)
    if state.get("nudge_mode") and actual_name:
        state["nudge_overlay"] = actual_name
        set_nudge_message("")
    return entry


def clear_active_overlays():
    state["active_overlays"] = []
    state["current_idx"] = 0
    state["nudge_overlay"] = None
    set_nudge_message("")


def arm_snapshot_wait():
    state['prod_waiting'] = True
    state['prod_frozen'] = False
    state['prod_frame'] = None
    return True


def capture_loop():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    ensure_snapshot_dir()
    reload_overlays()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if state.get('mode') == 'dev':
            state['show_face_box'] = True

        # FPS calculation
        now = time.time()
        prev = state.get("_prev_time", None)
        if prev is None:
            state["_prev_time"] = now
        else:
            dt = now - prev
            if dt > 0:
                state["fps"] = round(1.0 / dt, 1)
            state["_prev_time"] = now

        # If in prod mode and frozen, skip capture processing and keep prod_frame
        if state.get('mode') == 'prod' and state.get('prod_frozen'):
            frozen = state.get('prod_frame')
            if frozen is not None:
                state['frame'] = frozen
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # choose best face by combined area+center heuristic
        from face1 import select_best_face

        chosen = select_best_face(faces, frame.shape[1], frame.shape[0])

        if chosen is not None:
            x, y, w, h = chosen
            if state["show_face_box"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), state["face_box_thickness"])

            active_names = list(state.get("active_overlays", []))
            if active_names:
                for overlay_name in active_names:
                    entry = get_overlay_entry(overlay_name)
                    if entry is None:
                        continue
                    img = entry["image"]
                    cfg = entry.get("config", {})

                    scale = cfg.get("scale", 0.9)
                    y_offset = cfg.get("y_offset", 0.55)
                    x_anchor = cfg.get("x_anchor", 0.5)

                    overlay_w = int(w * scale)
                    aspect = img.shape[0] / img.shape[1]
                    overlay_h = int(overlay_w * aspect)

                    offset_x_pct = float(cfg.get("offset_x", 0.0))
                    offset_y_pct = float(cfg.get("offset_y", 0.0))
                    offset_x_px = int(round((offset_x_pct / 100.0) * (w / 2.0)))
                    offset_y_px = int(round((offset_y_pct / 100.0) * (h / 2.0)))

                    overlay_x = int(x + (w * x_anchor) - (overlay_w * x_anchor) + offset_x_px)
                    overlay_y = int(y + int(h * y_offset) + offset_y_px)

                    frame = overlay_transparent(frame, img, overlay_x, overlay_y, (overlay_w, overlay_h))

            # if we're in prod mode and waiting for a snapshot, freeze this frame now
            if state.get('mode') == 'prod' and state.get('prod_waiting') and not state.get('prod_frozen'):
                frozen_frame = frame.copy()
                state['prod_frame'] = frozen_frame
                state['prod_frozen'] = True
                state['prod_waiting'] = False
                save_snapshot(frozen_frame)

        # status
        if state.get('mode') != 'prod':
            active_names = state.get("active_overlays", [])
            if active_names:
                display = ", ".join(name.rsplit('.', 1)[0] for name in active_names)
                status = f"Overlays: {display}"
            else:
                status = "Overlays: none"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        state["frame"] = frame
        time.sleep(0.01)

    cap.release()


@app.route('/api/set_mode', methods=['POST'])
def api_set_mode():
    m = request.args.get('mode', request.form.get('mode', 'dev'))
    if m not in ('dev', 'prod'):
        return jsonify(success=False, error='invalid mode'), 400
    state['mode'] = m
    # changing modes clears any frozen/waiting prod state so live stream resumes
    state['prod_frozen'] = False
    state['prod_frame'] = None
    state['prod_waiting'] = False
    if m == 'dev':
        state['show_face_box'] = True
    else:
        state['show_face_box'] = False
    return jsonify(success=True, mode=m)


@app.route('/api/prod_freeze', methods=['POST'])
def api_prod_freeze():
    # direct freeze (manual): freeze the current frame immediately
    if state.get('frame') is None:
        return jsonify(success=False, error='no frame'), 400
    state['prod_frame'] = state['frame'].copy()
    state['prod_frozen'] = True
    state['prod_waiting'] = False
    return jsonify(success=True)


@app.route('/api/prod_reset', methods=['POST'])
def api_prod_reset():
    # reset to waiting-for-snapshot state
    state['prod_frozen'] = False
    state['prod_frame'] = None
    state['prod_waiting'] = False
    return jsonify(success=True)


@app.route('/api/prod_snapshot', methods=['POST'])
def api_prod_snapshot():
    if state.get('mode') != 'prod':
        return jsonify(success=False, error='snapshot only available in prod mode', mode=state.get('mode')), 409
    if state.get('prod_waiting'):
        return jsonify(success=True, waiting=True, mode=state.get('mode'))
    arm_snapshot_wait()
    return jsonify(success=True, waiting=True, mode=state.get('mode'))


@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    if state.get('mode') != 'prod':
        return jsonify(success=False, error='snapshot only available in prod mode', mode=state.get('mode')), 409
    if state.get('prod_waiting'):
        return jsonify(success=True, waiting=True, mode=state.get('mode'))
    arm_snapshot_wait()
    return jsonify(success=True, waiting=True, mode=state.get('mode'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/MeltedMonster-ARPLA.ttf')
def serve_melted_monster_font():
    base_dir = os.path.dirname(__file__)
    return send_from_directory(base_dir, 'MeltedMonster-ARPLA.ttf', mimetype='font/ttf')


def gen_frames():
    while True:
        frame = state.get("frame")
        if frame is None:
            time.sleep(0.01)
            continue
        # encode as jpeg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/toggle_overlay', methods=['POST'])
def api_toggle_overlay():
    if state.get('active_overlays'):
        clear_active_overlays()
    return jsonify(success=True, overlay=None, active_overlays=state.get('active_overlays', []))


@app.route('/api/next_overlay', methods=['POST'])
def api_next_overlay():
    overlays = state.get('overlays') or []
    added_name = None
    if overlays:
        state['current_idx'] = (state.get('current_idx', -1) + 1) % len(overlays)
        entry = overlays[state['current_idx']]
        added = add_overlay_by_name(entry.get('name'))
        if isinstance(added, dict):
            added_name = added.get('name')
    return jsonify(success=True, overlay=added_name, active_overlays=state.get('active_overlays', []))


@app.route('/api/prev_overlay', methods=['POST'])
def api_prev_overlay():
    overlays = state.get('overlays') or []
    added_name = None
    if overlays:
        state['current_idx'] = (state.get('current_idx', 0) - 1) % len(overlays)
        entry = overlays[state['current_idx']]
        added = add_overlay_by_name(entry.get('name'))
        if isinstance(added, dict):
            added_name = added.get('name')
    return jsonify(success=True, overlay=added_name, active_overlays=state.get('active_overlays', []))


@app.route('/api/overlay/reset', methods=['POST'])
def api_reset_overlays():
    clear_active_overlays()
    return jsonify(success=True, overlay=None, active_overlays=state.get('active_overlays', []))


@app.route('/api/overlay/<overlay_name>', methods=['POST'])
def api_set_overlay_name(overlay_name):
    overlay = add_overlay_by_name(overlay_name)
    if overlay is None:
        return jsonify(
            success=False,
            error='overlay not found',
            requested=overlay_name,
            available=list_overlay_names(),
        ), 404
    return jsonify(
        success=True,
        overlay=overlay.get('name') if isinstance(overlay, dict) else overlay,
        active_overlays=state.get('active_overlays', []),
    )


@app.route('/api/toggle_facebox', methods=['POST'])
def api_toggle_facebox():
    if state.get('mode') == 'dev':
        return jsonify(success=False, error='face box always on in dev mode'), 409
    state['show_face_box'] = not state['show_face_box']
    return jsonify(success=True, show_face_box=state['show_face_box'])


@app.route('/api/toggle_nudge', methods=['POST'])
def api_toggle_nudge():
    state['nudge_mode'] = not state.get('nudge_mode', False)
    if state['nudge_mode']:
        target = ensure_nudge_target()
        if target:
            set_nudge_message("")
        else:
            set_nudge_message("Activate an overlay to nudge.")
    else:
        state['nudge_overlay'] = None
        set_nudge_message("")
    target = state.get('nudge_overlay')
    return jsonify(
        success=True,
        nudge=state['nudge_mode'],
        target=target,
        config=get_overlay_config_snapshot(target) if target else None,
        message=state.get('nudge_message', ''),
        nudge_message=state.get('nudge_message', ''),
    )


@app.route('/api/nudge/<action>', methods=['POST'])
def api_nudge_action(action):
    if not state.get('nudge_mode'):
        set_nudge_message('Enable nudge mode first.')
        return jsonify(
            success=False,
            nudge=False,
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 409

    target = ensure_nudge_target()
    if not target:
        set_nudge_message('Select an overlay to edit.')
        return jsonify(
            success=False,
            nudge=True,
            target=None,
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 409

    entry = get_overlay_entry(target)
    if entry is None:
        set_nudge_message('Overlay not available; reload overlays.')
        return jsonify(
            success=False,
            nudge=True,
            target=None,
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 404

    cfg = entry.setdefault('config', {})
    action = (action or '').lower()

    def extract_numeric_value():
        data = request.get_json(silent=True)
        if isinstance(data, dict) and 'value' in data:
            raw = data['value']
        else:
            raw = request.form.get('value', request.args.get('value'))
        if raw is None or raw == '':
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    offset_step = 5.0
    scale_step = 0.02
    anchor_step = 0.01
    y_step = 0.01

    handled = True
    if action == 'offset_up':
        cfg['offset_y'] = round(float(cfg.get('offset_y', 0.0)) - offset_step, 3)
        set_nudge_message(f"offset_y={cfg['offset_y']}%")
    elif action == 'offset_down':
        cfg['offset_y'] = round(float(cfg.get('offset_y', 0.0)) + offset_step, 3)
        set_nudge_message(f"offset_y={cfg['offset_y']}%")
    elif action == 'offset_left':
        cfg['offset_x'] = round(float(cfg.get('offset_x', 0.0)) - offset_step, 3)
        set_nudge_message(f"offset_x={cfg['offset_x']}%")
    elif action == 'offset_right':
        cfg['offset_x'] = round(float(cfg.get('offset_x', 0.0)) + offset_step, 3)
        set_nudge_message(f"offset_x={cfg['offset_x']}%")
    elif action == 'set_offset_x':
        value = extract_numeric_value()
        if value is None:
            set_nudge_message('Provide numeric value for offset_x.')
            return jsonify(
                success=False,
                nudge=True,
                target=target,
                config=get_overlay_config_snapshot(target),
                message=state.get('nudge_message', ''),
                nudge_message=state.get('nudge_message', ''),
            ), 400
        cfg['offset_x'] = round(float(value), 3)
        set_nudge_message(f"offset_x={cfg['offset_x']}%")
    elif action == 'set_offset_y':
        value = extract_numeric_value()
        if value is None:
            set_nudge_message('Provide numeric value for offset_y.')
            return jsonify(
                success=False,
                nudge=True,
                target=target,
                config=get_overlay_config_snapshot(target),
                message=state.get('nudge_message', ''),
                nudge_message=state.get('nudge_message', ''),
            ), 400
        cfg['offset_y'] = round(float(value), 3)
        set_nudge_message(f"offset_y={cfg['offset_y']}%")
    elif action == 'scale_up':
        cfg['scale'] = round(float(cfg.get('scale', 0.9)) + scale_step, 3)
        set_nudge_message(f"scale={cfg['scale']}")
    elif action == 'scale_down':
        cfg['scale'] = round(max(0.01, float(cfg.get('scale', 0.9)) - scale_step), 3)
        set_nudge_message(f"scale={cfg['scale']}")
    elif action == 'anchor_left':
        cfg['x_anchor'] = round(max(0.0, float(cfg.get('x_anchor', 0.5)) - anchor_step), 3)
        set_nudge_message(f"x_anchor={cfg['x_anchor']}")
    elif action == 'anchor_right':
        cfg['x_anchor'] = round(min(1.0, float(cfg.get('x_anchor', 0.5)) + anchor_step), 3)
        set_nudge_message(f"x_anchor={cfg['x_anchor']}")
    elif action == 'y_offset_up':
        cfg['y_offset'] = round(float(cfg.get('y_offset', 0.55)) - y_step, 3)
        set_nudge_message(f"y_offset={cfg['y_offset']}")
    elif action == 'y_offset_down':
        cfg['y_offset'] = round(float(cfg.get('y_offset', 0.55)) + y_step, 3)
        set_nudge_message(f"y_offset={cfg['y_offset']}")
    elif action == 'save':
        ok = persist_overlays_config()
        set_nudge_message('Overlay config saved.' if ok else 'Failed to save overlay config.')
    elif action == 'reload':
        reload_overlays(force=True)
        target = ensure_nudge_target()
        if not target:
            return jsonify(
                success=True,
                nudge=True,
                target=None,
                config=None,
                message=state.get('nudge_message', ''),
                nudge_message=state.get('nudge_message', ''),
            )
        set_nudge_message('Reloaded overlays config.')
        entry = get_overlay_entry(target)
        cfg = entry.get('config', {}) if entry else {}
    else:
        handled = False

    if not handled:
        set_nudge_message(f'Unknown nudge action: {action}')
        return jsonify(
            success=False,
            nudge=True,
            target=target,
            config=get_overlay_config_snapshot(target),
            message=state.get('nudge_message', ''),
            nudge_message=state.get('nudge_message', ''),
        ), 400

    return jsonify(
        success=True,
        nudge=True,
        target=target,
        config=get_overlay_config_snapshot(target),
        message=state.get('nudge_message', ''),
        nudge_message=state.get('nudge_message', ''),
    )


@app.route('/api/story', methods=['GET', 'POST'])
def api_story():
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if isinstance(data, dict) and 'text' in data:
            raw_text = data.get('text')
        else:
            raw_text = request.form.get('text', request.args.get('text'))

        if raw_text is None:
            return jsonify(success=False, error='Missing story text.'), 400
        text = str(raw_text).strip()
        state['story_text'] = text
        state['story_updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        return jsonify(success=True, text=state['story_text'], updated=state['story_updated'])

    return jsonify(
        text=state.get('story_text', ''),
        updated=state.get('story_updated'),
    )


@app.route('/api/update_story', methods=['POST'])
def api_update_story():
    """Update story based on user inputs (name, costume, treat)."""
    data = request.get_json(silent=True)
    debug_log_api_call('/api/update_story', 'POST', data)
    
    if not isinstance(data, dict):
        debug_log("Invalid JSON data received", "ERROR")
        return jsonify(success=False, error='Invalid JSON data.'), 400
    
    # Extract user inputs
    name = data.get('name', '').strip()
    costume = data.get('costume', '').strip()
    treat = data.get('treat', '').strip()
    
    debug_log(f"Updating story with inputs: name='{name}', costume='{costume}', treat='{treat}'")
    
    # Update state
    state['user_name'] = name
    state['user_costume'] = costume
    state['user_treat'] = treat
    
    # Generate new story
    story = generate_story(name, costume, treat)
    state['story_text'] = story
    state['story_updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    
    debug_log(f"Story generated successfully: {story[:100]}...")
    
    return jsonify(
        success=True, 
        story=story,
        updated=state['story_updated']
    )


@app.route('/api/prompt_config', methods=['GET', 'POST'])
def api_prompt_config():
    """Get or update prompt configuration."""
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(success=False, error='Invalid JSON data.'), 400
        
        # Update prompt configuration
        global prompt_config
        if prompt_config is None:
            prompt_config = {}
        
        # Remove legacy prompt fields so the in-code templates remain authoritative.
        for legacy_key in ('system_prompt', 'main_prompt_template', 'sample_templates'):
            data.pop(legacy_key, None)
            prompt_config.pop(legacy_key, None)

        if 'llm_parameters' in data:
            prompt_config['llm_parameters'] = data['llm_parameters']
        if 'fallback_templates' in data:
            prompt_config['fallback_templates'] = data['fallback_templates']
        
        # Save configuration
        if save_prompt_config():
            return jsonify(success=True, config=prompt_config)
        else:
            return jsonify(success=False, error='Failed to save configuration.'), 500
    
    # GET request - return current configuration
    return jsonify(
        success=True,
        config=prompt_config or {}
    )


@app.route('/api/prompt_config/reset', methods=['POST'])
def api_reset_prompt_config():
    """Reset prompt configuration to defaults."""
    global prompt_config, debug_config
    config_path = os.path.join(os.path.dirname(__file__), "prompt_config.json")
    
    # Load default configuration
    default_config = {
        "llm_parameters": {
            "max_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.9,
            "stop": ["\n\n", "###", "---"]
        },
        "debug_mode": {
            "enabled": False,
            "log_level": "INFO",
            "show_prompts": True,
            "show_llm_parameters": True,
            "show_story_generation": True,
            "show_api_calls": True,
            "show_errors": True
        },
        "fallback_templates": [
            "{name} haunts the office in a {costume}, lurking near the {treat}. Employees scream!",
            "Dressed as a {costume}, {name} casts a spooky shadow while reaching for a {treat}."
        ]
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        prompt_config = default_config
        debug_config = default_config.get('debug_mode', {})
        return jsonify(success=True, config=prompt_config)
    except Exception as e:
        return jsonify(success=False, error=f'Failed to reset configuration: {e}'), 500


@app.route('/api/debug_mode', methods=['GET', 'POST'])
def api_debug_mode():
    """Get or update debug mode configuration."""
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(success=False, error='Invalid JSON data.'), 400
        
        # Update debug configuration
        global debug_config, prompt_config
        if prompt_config and 'debug_mode' in prompt_config:
            if 'enabled' in data:
                prompt_config['debug_mode']['enabled'] = bool(data['enabled'])
            if 'log_level' in data:
                prompt_config['debug_mode']['log_level'] = data['log_level']
            if 'show_prompts' in data:
                prompt_config['debug_mode']['show_prompts'] = bool(data['show_prompts'])
            if 'show_llm_parameters' in data:
                prompt_config['debug_mode']['show_llm_parameters'] = bool(data['show_llm_parameters'])
            if 'show_story_generation' in data:
                prompt_config['debug_mode']['show_story_generation'] = bool(data['show_story_generation'])
            if 'show_api_calls' in data:
                prompt_config['debug_mode']['show_api_calls'] = bool(data['show_api_calls'])
            if 'show_errors' in data:
                prompt_config['debug_mode']['show_errors'] = bool(data['show_errors'])
            
            debug_config = prompt_config['debug_mode']
            
            # Save configuration
            if save_prompt_config():
                return jsonify(success=True, debug_config=debug_config)
            else:
                return jsonify(success=False, error='Failed to save debug configuration.'), 500
        
        return jsonify(success=False, error='Debug configuration not found.'), 404
    
    # GET request - return current debug configuration
    return jsonify(
        success=True,
        debug_config=debug_config or {}
    )


@app.route('/api/debug_mode/toggle', methods=['POST'])
def api_toggle_debug_mode():
    """Toggle debug mode on/off."""
    global debug_config, prompt_config
    
    if not prompt_config or 'debug_mode' not in prompt_config:
        return jsonify(success=False, error='Debug configuration not found.'), 404
    
    # Toggle debug mode
    current_state = prompt_config['debug_mode'].get('enabled', False)
    prompt_config['debug_mode']['enabled'] = not current_state
    debug_config = prompt_config['debug_mode']
    
    # Save configuration
    if save_prompt_config():
        new_state = "ENABLED" if debug_config['enabled'] else "DISABLED"
        debug_log(f"Debug mode {new_state}")
        return jsonify(success=True, enabled=debug_config['enabled'], message=f"Debug mode {new_state}")
    else:
        return jsonify(success=False, error='Failed to save debug configuration.'), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    active = list(state.get('active_overlays', []))
    primary = active[-1] if active else None
    target = state.get('nudge_overlay') if state.get('nudge_mode') else None
    return jsonify(
        fps=state.get('fps', 0.0),
        overlay=primary,
        overlays=list_overlay_names(),
        active_overlays=active,
        nudge=state.get('nudge_mode', False),
        nudge_target=target,
        nudge_config=get_overlay_config_snapshot(target) if target else None,
        nudge_message=state.get('nudge_message', ''),
        mode=state.get('mode','prod'),
        prod_frozen=state.get('prod_frozen', False),
        prod_waiting=state.get('prod_waiting', False),
        last_snapshot=state.get('last_snapshot_path'),
        story_text=state.get('story_text', ''),
        story_updated=state.get('story_updated'),
        user_name=state.get('user_name', ''),
        user_costume=state.get('user_costume', ''),
        user_treat=state.get('user_treat', ''),
    )


if __name__ == '__main__':
    # Load prompt configuration
    print("=" * 60)
    print("Starting Pumpkin AI Server")
    print("=" * 60)
    
    print("\n[1/3] Loading prompt configuration...")
    load_prompt_config()
    
    print(f"\n[2/3] Initializing LLM...")
    print(f"Ollama available: {OLLAMA_AVAILABLE}")
    print(f"llama.cpp available: {LLAMA_CPP_AVAILABLE}")
    
    llm_init_success = initialize_llm()
    if llm_init_success:
        print(f"✓ LLM initialized successfully using {llm_type}!")
    else:
        print("✗ LLM initialization failed - will use fallback story generation")
        print("Tip: Install Ollama (https://ollama.ai) or llama-cpp-python for LLM support")
    
    print(f"\n[3/3] Starting web server...")
    
    # start capture thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    port = int(os.environ.get('PUMPKIN_PORT', '5000'))
    
    print("\n" + "=" * 60)
    print(f"Server starting on http://0.0.0.0:{port}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port)
