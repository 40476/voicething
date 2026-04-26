#!/usr/bin/python3
import faulthandler
faulthandler.enable()
import os

# =============== HARDWARE FIX FOR AMD GPU ===============
# These MUST be set before importing torch or faster-whisper to prevent CTranslate2 segfaults
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
# Unleash the CPU threads!
os.environ["OMP_NUM_THREADS"] = "88"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import curses, time, librosa, queue, threading, numpy as np, textwrap, subprocess, signal, wave, tempfile, whisper, pyaudio, torch, json, resampy, contextlib, sys
from datetime import datetime
from pathlib import Path
from gtts import gTTS

try:
    import vosk
except ImportError:
    vosk = None

# =============== CONFIG PATHS ===============
CONFIG_DIR = Path.home() / ".local/share/voicething"
SOUNDS_DIR = CONFIG_DIR / "sounds"
MODELS_DIR = CONFIG_DIR / "models"
CONFIG_FILE = CONFIG_DIR / "voicething.json"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============== DEFAULT SETTINGS ===============
default_settings = {
    "whisper_model": "small",
    "whisper_language": "en",
    "whisper_backend": "local_openai_whisper", # Options: local_openai_whisper, vosk
    "whisper_device": "cpu",        
    "whisper_compute_type": "int8", 
    "voice_model": "en_US/hifi-tts_low",
    "audio_device": None,           # Microphone Input
    "loopback_device": None,        # System Audio Monitor (for AEC)
    "aec_enabled": True,            # Internal Algorithmic Echo Cancellation
    "aec_aggressiveness": 2.0,      # Multiplier for subtraction (higher = more aggressive)
    "tts_enabled": True,
    "prosody_enabled": True,
    "sound_emojis": {},
    "use_python_tts": False,
    "vox_enabled": True,            
    "vox_threshold": 0.015,         
    "vox_silence_duration": 0.5,    
    "echo_cancellation_delay": 0.5, # App-level TTS ducking safety net
}

# Load or create config
if CONFIG_FILE.exists():
    try:
        settings = json.loads(CONFIG_FILE.read_text())
        for key in default_settings:
            if key not in settings:
                settings[key] = default_settings[key]
    except:
        settings = default_settings.copy()
        CONFIG_FILE.write_text(json.dumps(settings, indent=2))
else:
    settings = default_settings.copy()
    CONFIG_FILE.write_text(json.dumps(settings, indent=2))

setting_keys = list(settings.keys())
selected_index = 0
settings_scroll = 0
should_run = True
torch.backends.cudnn.enabled = False

# =============== SOUNDBOARD STATE ===============
soundboard_visible = True
sound_files = []
sound_selected = 0
sound_scroll = 0
last_play_time = 0

# =============== STATE ===============
transcript_log = []
live_partial_text = "" # For storing live streaming partial results
log_lines = []
log_scroll = 0
MAX_LOG_LINES = 100
audio_q = queue.Queue()
log_q = queue.Queue()
config_mtime = 0  # Track config file modification time

# --- TTS PIPELINE QUEUES ---
tts_text_queue = queue.Queue()  # Holds (text, prosody_params)
tts_audio_queue = queue.Queue() # Holds raw wav bytes

child_procs = []
model_ready = threading.Event()
model = None
engine = None  # Global engine reference (Vosk or WLK)
download_status = "" # For tracking download progress

# Global PyAudio instance to prevent termination segfaults
global_pa = None

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# =============== ECHO CANCELLATION STATE ===============
active_playbacks = 0
playback_lock = threading.Lock()
last_playback_end = 0.0

def start_playback():
    global active_playbacks
    with playback_lock:
        active_playbacks += 1

def stop_playback():
    global active_playbacks, last_playback_end
    with playback_lock:
        active_playbacks -= 1
        if active_playbacks <= 0:
            active_playbacks = 0
            last_playback_end = time.time()

# ================ SOUND PLAYBACK ================
def play_sound(filename):
    global last_play_time
    current_time = time.time()
    
    if current_time - last_play_time < 0.5:
        return
        
    last_play_time = current_time
    filepath = SOUNDS_DIR / filename
    
    def _play():
        start_playback()
        try:
            subprocess.run([
                "paplay", "--device=TTS_voice", "--volume=65536", str(filepath)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"[sound] Error: {e}")
        finally:
            stop_playback()

    threading.Thread(target=_play, daemon=True).start()

# =============== HELPERS ===============
def handle_mouse_event(win):
    try:
        _, mx, my, _, btn_state = curses.getmouse()
        if not (btn_state & curses.BUTTON1_PRESSED): return None

        win_start_y, win_start_x = win.getbegyx()
        win_height, win_width = win.getmaxyx()

        if not (win_start_y <= my < win_start_y + win_height and
                win_start_x <= mx < win_start_x + win_width):
            return None

        rel_y = my - win_start_y
        if rel_y < 1 or rel_y >= win_height - 1: return None

        button_row = (rel_y - 1) // 3  
        sound_index = sound_scroll + button_row

        if 0 <= sound_index < len(sound_files):
            return sound_index
        return None
    except Exception as e:
        log(f"[Mouse] Error: {e}")
        return None

def save_config():
    CONFIG_FILE.write_text(json.dumps(settings, indent=2))

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_q.put(f"[{timestamp}] {msg}")

# =============== AUDIO & TTS PIPELINE ===============
def play_tone(frequency=440, duration=0.05, sink="TTS_voice"):
    try:
        cmd = f"sox -n -r 44100 -c 2 -b 16 -t raw - synth {duration} sine {frequency}"
        if sink:
            cmd += f" | pacat --client-name=TonePlayer --device={sink}"
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"[tone] Error: {e}")

def generate_python_tts(text):
    """Synchronous gTTS generation and playback to replace the infinite loop worker."""
    try:
        log(f"[Python-TTS] Generating: {text[:30]}...")
        tts = gTTS(text=text, lang=settings.get("whisper_language", "en"))
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            
            start_playback() # Mute mic via ducking safety net
            subprocess.run(["mpg123", f.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stop_playback()  
            
            os.unlink(f.name)
    except Exception as e:
        log(f"[Python-TTS] Error: {e}")

# --- TTS WORKER USING SUBPROCESS ---
def tts_subprocess_worker():
    """Pulls text, formats for generation, pushes audio to playback."""
    log("[TTS-Sub] Worker started")

    while should_run:
        try:
            item = tts_text_queue.get(timeout=1)
            text, prosody = item

            if not settings.get("tts_enabled", True):
                tts_text_queue.task_done()
                continue

            # Route to correct TTS engine
            if settings.get("use_python_tts", False):
                generate_python_tts(text)
                tts_text_queue.task_done()
                continue

            # Build mimic3_tts command with parameters
            cmd = [sys.executable, "-m", "mimic3_tts"]
            cmd.extend(["--voice", settings["voice_model"]])

            if prosody and settings.get("prosody_enabled", True):
                length_scale, noise_scale, noise_w = map_prosody_to_mimic3(prosody)
                cmd.extend(["--noise-scale", str(noise_scale)])
                cmd.extend(["--length-scale", str(length_scale)])
                cmd.extend(["--noise-w", str(noise_w)])

            try:
                log(f"[TTS-Sub] Generating: {text[:30]}...")
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                child_procs.append(proc)

                try:
                    proc.stdin.write((text + "\n").encode('utf-8'))
                    proc.stdin.close()
                except BrokenPipeError:
                    log("[TTS-Sub] Broken pipe, process may have crashed")
                    tts_text_queue.task_done()
                    continue

                wav_data = proc.stdout.read()
                proc.wait()

                if proc in child_procs:
                    child_procs.remove(proc)

                if proc.returncode != 0:
                    err = proc.stderr.read().decode('utf-8', errors='ignore')
                    log(f"[TTS-Sub] Error: {err[:100]}")
                elif len(wav_data) > 0:
                    tts_audio_queue.put(wav_data)
                    log(f"[TTS-Sub] Got {len(wav_data)} bytes")
                else:
                    log("[TTS-Sub] Empty output")

            except Exception as e:
                log(f"[TTS-Sub] Request failed: {e}")

            tts_text_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            log(f"[TTS-Sub] Error: {e}")


def tts_playback_worker():
    """Pulls audio bytes, plays them via pacat."""
    log("[TTS-Play] Worker started")
    while should_run:
        try:
            wav_data = tts_audio_queue.get(timeout=1)
            log("[TTS-Play] Playing audio...")
            
            start_playback() # Mute mic via ducking safety net
            try:
                pacat = subprocess.Popen(
                    ["pacat", "--client-name=VoiceThingTTS", "--device=TTS_voice", 
                     "--format=s16le", "--rate=22050", "--channels=1"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                pacat.communicate(input=wav_data)
            finally:
                stop_playback() 
                
            tts_audio_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            log(f"[TTS-Play] Error: {e}")


# =============== RESTORED UI FUNCTIONS ===============
def update_transcript(win):
    global live_partial_text
    win.clear()
    win.box()
    win.addstr(0, 2, " TRANSCRIPT ")
    max_y, max_x = win.getmaxyx()
    
    # Leave room for the live partial text at the bottom
    visible_lines = max_y - 3 
    if visible_lines <= 0: return

    display_lines = transcript_log[-visible_lines:]
    
    for i, line in enumerate(display_lines):
        try:
            win.addstr(i + 1, 1, f" {line[:max_x - 4]} ")
        except curses.error:
            pass
            
    # Draw live partial text
    if live_partial_text:
        try:
            win.addstr(max_y - 2, 1, f"> {live_partial_text[:max_x - 4]}", curses.color_pair(2))
        except curses.error:
            pass

    win.refresh()

def update_soundboard(win):
    win.clear()
    win.box()
    win.addstr(0, 2, " SOUNDBOARD (Press 's' to hide) ")
    max_y, max_x = win.getmaxyx()
    
    visible_rows = (max_y - 2) // 3
    if visible_rows <= 0: return

    global sound_scroll
    if sound_selected < sound_scroll:
        sound_scroll = sound_selected
    elif sound_selected >= sound_scroll + visible_rows:
        sound_scroll = sound_selected - visible_rows + 1

    for i in range(visible_rows):
        idx = sound_scroll + i
        if idx >= len(sound_files): break
            
        color = curses.color_pair(1 if idx == sound_selected else 2)
        y = 1 + (i * 3)
        try:
            win.attrset(color)
            win.addstr(y, 2, f"[{idx}] {sound_files[idx][:max_x-8]}")
            win.addstr(y+1, 2, "-" * (max_x - 4))
        except curses.error: pass
            
    win.refresh()

def update_settings(win):
    win.clear()
    win.box()
    win.addstr(0, 2, " SETTINGS ")
    max_y, max_x = win.getmaxyx()
    
    visible_lines = max_y - 2
    if visible_lines <= 0: return

    global settings_scroll
    if selected_index < settings_scroll:
        settings_scroll = selected_index
    elif selected_index >= settings_scroll + visible_lines:
        settings_scroll = selected_index - visible_lines + 1

    for i in range(visible_lines):
        idx = settings_scroll + i
        if idx >= len(setting_keys): break
        
        key = setting_keys[idx]
        val = settings[key]
        color = curses.color_pair(1 if idx == selected_index else 2)
        
        try:
            win.attrset(color)
            line = f"{key}: {val}"
            win.addstr(i + 1, 2, line[:max_x - 4])
        except curses.error: pass
    win.refresh()

def update_logs(win):
    while not log_q.empty():
        try:
            msg = log_q.get_nowait()
            log_lines.append(msg)
            if len(log_lines) > MAX_LOG_LINES:
                log_lines.pop(0)
        except queue.Empty: break

    win.clear()
    win.box()
    win.addstr(0, 2, " LOGS ")
    max_y, max_x = win.getmaxyx()
    
    visible_lines = max_y - 2
    if visible_lines <= 0: return
    
    display_lines = log_lines[-visible_lines:]
    
    for i, line in enumerate(display_lines):
        try: win.addstr(i + 1, 2, line[:max_x - 4])
        except curses.error: pass
    win.refresh()

def update_command(win):
    win.clear()
    win.box()
    win.addstr(0, 2, " COMMANDS ")
    try:
        win.addstr(1, 2, "Up/Down: Navigate | Enter: Select/Edit | s: Toggle Soundboard | q: Quit")
        if download_status:
            win.addstr(2, 2, f"DL: {download_status[:win.getmaxyx()[1]-8]}")
    except curses.error: pass
    win.refresh()

def edit_setting(win):
    key = setting_keys[selected_index]
    val = settings[key]
    
    if isinstance(val, bool):
        settings[key] = not val
        save_config()
        log(f"[config] {key} -> {settings[key]}")
    elif isinstance(val, float):
        # Quick toggle adjustments for float settings like vox_threshold
        if key == "vox_threshold":
            settings[key] = round(val + 0.005 if val < 0.1 else 0.005, 3)
        elif key == "vox_silence_duration":
            settings[key] = round(val + 0.5 if val < 3.0 else 0.5, 1)
        elif key == "aec_aggressiveness":
            settings[key] = round(val + 0.5 if val < 10.0 else 0.5, 1)
        elif key == "echo_cancellation_delay":
            settings[key] = round(val + 0.1 if val < 2.0 else 0.1, 1)
        save_config()
        log(f"[config] {key} -> {settings[key]}")
    elif key in ["whisper_device", "whisper_compute_type", "whisper_backend"]:
        # Allow toggling between valid options
        if key == "whisper_device":
            settings[key] = "cuda" if val == "cpu" else "cpu"
        elif key == "whisper_compute_type":
            settings[key] = "float16" if val == "int8" else "int8"
        elif key == "whisper_backend":
            backends = ["local_openai_whisper", "vosk"]
            current_idx = backends.index(val) if val in backends else 0
            settings[key] = backends[(current_idx + 1) % len(backends)]
        save_config()
        log(f"[config] {key} -> {settings[key]}. Restart required.")
    elif key in ["audio_device", "loopback_device"]:
        devices = list_audio_devices()
        if devices:
            dev_list = [f"{d[1]}" for d in devices]
            idx = device_selector(win, dev_list)
            if idx is not None:
                settings[key] = devices[idx][0]
                save_config()
                log(f"[config] {key} set to {devices[idx][1]}")

def edit_sound_emoji(idx):
    if 0 <= idx < len(sound_files):
        log(f"[emoji] Select emoji functionality not implemented for {sound_files[idx]}")

def list_audio_devices():
    # Use the global PyAudio instance, do NOT terminate it.
    input_devices = []
    if global_pa is None: return input_devices
    
    for i in range(global_pa.get_device_count()):
        try:
            info = global_pa.get_device_info_by_index(i)
            # Find input devices (includes physical mics AND monitor loopbacks on PulseAudio)
            if info["maxInputChannels"] > 0:
                input_devices.append((i, info["name"]))
        except: pass
    return input_devices

def device_selector(win, devices):
    selected_index = 0
    scroll_offset = 0
    win.keypad(True)

    while True:
        win.clear()
        win.box()
        win.addstr(0, 2, " SELECT AUDIO DEVICE ")
        h, w = win.getmaxyx()
        visible_height = h - 2
        
        if selected_index < scroll_offset:
            scroll_offset = selected_index
        elif selected_index >= scroll_offset + visible_height:
            scroll_offset = selected_index - visible_height + 1

        for i in range(visible_height):
            idx = scroll_offset + i
            if idx >= len(devices): break
            line = f"{idx}: {devices[idx]}"
            color = curses.color_pair(1 if idx == selected_index else 2)
            win.attrset(color)
            win.addstr(i + 1, 2, line[:w - 4])

        win.refresh()
        key = win.getch()
        
        if key == curses.KEY_UP and selected_index > 0: selected_index -= 1
        elif key == curses.KEY_DOWN and selected_index < len(devices) - 1: selected_index += 1
        elif key in (10, 13): return selected_index
        elif key == 27: return None

# =============== AUDIO STREAMING & INTERNAL AEC ===============
def audio_stream_worker():
    mic_stream = None
    ref_stream = None
    
    previous_mic = None
    previous_ref = None

    # VOX state
    is_recording = False
    audio_buffer = []
    silence_chunks = 0

    while should_run:
        try:
            mic_id = settings.get("audio_device")
            ref_id = settings.get("loopback_device")
            aec_enabled = settings.get("aec_enabled", True)

            if mic_id is None:
                time.sleep(1)
                continue

            # Handle Microphone Stream Restarts
            if previous_mic != mic_id:
                previous_mic = mic_id
                if mic_stream:
                    mic_stream.stop_stream()
                    mic_stream.close()
                    mic_stream = None
            
            # Handle Loopback Stream Restarts
            if aec_enabled and ref_id is not None and previous_ref != ref_id:
                previous_ref = ref_id
                if ref_stream:
                    ref_stream.stop_stream()
                    ref_stream.close()
                    ref_stream = None

            # Open Microphone
            if mic_stream is None:
                try:
                    if global_pa is not None:
                        mic_info = global_pa.get_device_info_by_index(mic_id)
                        log(f"[audio] Mic: {mic_info['name']}")
                        mic_stream = global_pa.open(
                            format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK,
                            input_device_index=mic_id
                        )
                        mic_stream.start_stream()
                except Exception as e:
                    log(f"[audio] Mic init error: {e}")
                    time.sleep(2)
                    continue

            # Open Loopback Reference (for AEC)
            if aec_enabled and ref_id is not None and ref_stream is None:
                try:
                    if global_pa is not None:
                        ref_info = global_pa.get_device_info_by_index(ref_id)
                        log(f"[audio] Ref: {ref_info['name']}")
                        ref_stream = global_pa.open(
                            format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK,
                            input_device_index=ref_id
                        )
                        ref_stream.start_stream()
                except Exception as e:
                    log(f"[audio] Ref init error (AEC disabled): {e}")
                    ref_stream = None

            # Process Audio
            if mic_stream and mic_stream.is_active():
                mic_data = mic_stream.read(CHUNK, exception_on_overflow=False)
                mic_np = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32)
                
                clean_np = mic_np # Default to unaltered mic audio

                # ==== SPECTRAL SUBTRACTION AEC LOGIC ====
                if aec_enabled and ref_stream and ref_stream.is_active():
                    ref_data = ref_stream.read(CHUNK, exception_on_overflow=False)
                    ref_np = np.frombuffer(ref_data, dtype=np.int16).astype(np.float32)
                    
                    # 1. Fast Fourier Transform (Time Domain -> Frequency Domain)
                    mic_fft = np.fft.rfft(mic_np)
                    ref_fft = np.fft.rfft(ref_np)

                    # 2. Extract Magnitudes and Phase
                    mic_mag = np.abs(mic_fft)
                    ref_mag = np.abs(ref_fft)
                    mic_phase = np.angle(mic_fft)

                    # 3. Spectral Subtraction (Erase Discord/System audio from Mic)
                    aggressiveness = settings.get("aec_aggressiveness", 2.0)
                    clean_mag = mic_mag - (aggressiveness * ref_mag)
                    clean_mag = np.maximum(clean_mag, 0.0) # Prevent negative magnitudes

                    # 4. Inverse FFT (Frequency Domain -> Clean Time Domain)
                    clean_fft = clean_mag * np.exp(1j * mic_phase)
                    clean_np = np.fft.irfft(clean_fft).astype(np.float32)


                # ==== APP DUCKING SAFETY NET ====
                # We still keep the ducking lock just in case the TTS causes weird artifacts
                with playback_lock:
                    currently_playing = active_playbacks > 0
                
                delay = settings.get("echo_cancellation_delay", 0.5)
                if currently_playing or (time.time() - last_playback_end < delay):
                    if is_recording:
                        log("[VOX] Muted mic during app playback.")
                        is_recording = False
                        audio_buffer = []
                        silence_chunks = 0
                    continue # Drop frame
                
                # ==== STANDARD VOX PROCESSING ====
                backend = settings.get("whisper_backend", "local_openai_whisper")
                
                # Convert cleaned numpy array back to bytes for Vosk/Whisper
                clean_bytes = clean_np.astype(np.int16).tobytes()

                if backend == "vosk":
                    # Vosk wants continuous audio stream for real-time live partials
                    audio_q.put(clean_bytes)
                elif settings.get("vox_enabled", True):
                    # Calculate volume (RMS energy) of CLEANED signal
                    rms = np.sqrt(np.mean((clean_np / 32768.0)**2) + 1e-6)
                    threshold = settings.get("vox_threshold", 0.015)
                    max_silence_chunks = int((settings.get("vox_silence_duration", 0.5) * RATE) / CHUNK)
                    
                    if rms > threshold:
                        if not is_recording:
                            log("[VOX] Speech detected...")
                            is_recording = True
                        silence_chunks = 0
                        audio_buffer.append(clean_bytes)
                    elif is_recording:
                        silence_chunks += 1
                        audio_buffer.append(clean_bytes)
                        
                        if silence_chunks > max_silence_chunks:
                            log("[VOX] Silence detected. Sending to transcriber.")
                            is_recording = False
                            audio_q.put(b''.join(audio_buffer))
                            audio_buffer = []
                else:
                    audio_q.put(clean_bytes)
            else:
                time.sleep(0.1)

        except Exception as e:
            log(f"[audio] Loop error: {e}")
            if mic_stream:
                try: mic_stream.close()
                except: pass
                mic_stream = None
            if ref_stream:
                try: ref_stream.close()
                except: pass
                ref_stream = None
            time.sleep(1)
    
    if mic_stream: mic_stream.close()
    if ref_stream: ref_stream.close()

def map_range(val, in_min, in_max, out_min, out_max):
    if in_max == in_min: return out_min
    ratio = (val - in_min) / (in_max - in_min)
    ratio = max(0.0, min(1.0, ratio))
    return out_min + ratio * (out_max - out_min)

def map_prosody_to_mimic3(prosody):
    if not settings.get("prosody_enabled", True): return 1.0, 0.667, 0.8 
    raw_pitch = prosody.get("pitch", 150)
    raw_energy = prosody.get("energy", 0.05)
    noise_scale = map_range(raw_energy, 0.01, 0.1, 0.3, 0.9)
    noise_w = map_range(raw_pitch, 80, 300, 0.4, 1.0)
    length_scale = 1.0
    return length_scale, noise_scale, noise_w

class DownloadProgressBar:
    def write(self, s: str) -> int:
        global download_status
        s = s.strip()
        if s: download_status = s
        return len(s)
    def flush(self) -> None: pass


# =============== TRANSCRIBE ===============

def load_models_sync():
    global model, engine, download_status
    
    backend_choice = settings.get("whisper_backend", "local_openai_whisper")
    engine = None
    model = None
    
    if backend_choice == "vosk":
        if vosk is None:
            log("[model] ERROR: Vosk is not installed. Please run: pip install vosk")
            download_status = "Error: pip install vosk required"
            model_ready.set()
            return
            
        try:
            log("[model] Loading Vosk streaming model...")
            # Automatically downloads the lightweight language model if not present
            vosk.SetLogLevel(-1) # Hide annoying logs
            model = vosk.Model(lang="en-us")
            engine = vosk.KaldiRecognizer(model, RATE)
            log("[model] Vosk live streaming ready.")
        except Exception as e:
            log(f"[model] Vosk init failed: {e}")
            download_status = f"Vosk Error: {str(e)[:40]}"
            
    else:
        log("[model] Loading native OpenAI Whisper...")
        try:
            with contextlib.redirect_stderr(DownloadProgressBar()):
                model = whisper.load_model(
                    settings["whisper_model"], 
                    download_root=str(MODELS_DIR),
                    device=settings.get("whisper_device", "cpu")
                )
            log("[model] Native Whisper ready.")
        except Exception as fallback_e:
            log(f"[model] Native Whisper load failed: {fallback_e}")
            download_status = f"Error: {str(fallback_e)[:40]}"

    model_ready.set()
    download_status = ""


def transcribe_worker():
    global engine, model, live_partial_text
    log("[transcribe] Worker active")
    backend_choice = settings.get("whisper_backend", "local_openai_whisper")

    while should_run:
        try:
            # Get audio from the queue
            audio_bytes = audio_q.get(timeout=1)
            
            transcription_text = ""
            
            if backend_choice == "vosk" and engine is not None:
                # VOSK TRUE REAL-TIME STREAMING
                if engine.AcceptWaveform(audio_bytes):
                    # Finalized sentence
                    res = json.loads(engine.Result())
                    transcription_text = res.get("text", "")
                    live_partial_text = "" # Clear the partial
                else:
                    # Partial sentence (user is still talking)
                    res = json.loads(engine.PartialResult())
                    partial = res.get("partial", "")
                    if partial:
                        live_partial_text = partial
            
            elif backend_choice == "local_openai_whisper" and model is not None:
                # STANDARD WHISPER CHUNK PROCESSING
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                res = model.transcribe(audio_np, language=settings.get("whisper_language", "en"))
                transcription_text = res.get("text", "")
            
            if transcription_text and transcription_text.strip():
                text = transcription_text.strip()
                timestamp = datetime.now().strftime("[%H:%M:%S]")
                log(f"[Rec] {text}")
                transcript_log.append(f"{timestamp} {text}")
                tts_text_queue.put((text, None))

            audio_q.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            log(f"[transcribe] Error: {e}")
            time.sleep(1)

def main(stdscr):
    global selected_index, log_scroll, should_run, config_mtime
    global soundboard_visible, sound_selected, sound_files
    global global_pa
    
    # Initialize PyAudio globally on the main thread ONLY ONCE
    global_pa = pyaudio.PyAudio()

    def final_cleanup():
        global should_run
        should_run = False
        cleanup_children()
        save_config()
        log("Shutting down...")
        time.sleep(0.2)
        if global_pa:
            global_pa.terminate()
    
    def cleanup_children():
        for p in child_procs:
            try: 
                p.terminate()
                p.wait(timeout=1)
            except: pass

    def check_config_reload():
        global config_mtime, settings, setting_keys
        try:
            current_mtime = CONFIG_FILE.stat().st_mtime
            if current_mtime > config_mtime:
                config_mtime = current_mtime
                with open(CONFIG_FILE, 'r') as f:
                    new_settings = json.load(f)
                for key in default_settings:
                    if key not in new_settings:
                        new_settings[key] = default_settings[key]
                settings.update(new_settings)
                setting_keys = list(settings.keys())
                log("[config] Reloaded settings from file")
        except Exception:
            pass 

    signal.signal(signal.SIGINT, lambda s, f: final_cleanup())
    
    try: config_mtime = CONFIG_FILE.stat().st_mtime
    except: config_mtime = 0
    
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    
    h, w = stdscr.getmaxyx()
    half = w // 2
    
    settings_win = curses.newwin(12, half, 0, 0)
    log_win = curses.newwin(h - 17, half, 12, 0)
    cmd_win = curses.newwin(5, half, h - 5, 0)
    transcript_win = curses.newwin(h - 2, w - half, 0, half)
    soundboard_win = None
    
    try: sound_files = sorted([f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.wav', '.mp3'))])
    except: sound_files = []
    
    # Draw loading message to avoid confusing the user while it blocks
    cmd_win.clear()
    cmd_win.box()
    cmd_win.addstr(1, 2, " LOADING AI MODELS... PLEASE WAIT. ")
    cmd_win.refresh()

    # Load models synchronously in the main thread to prevent C++ / CTranslate2 segfaults
    load_models_sync()

    # START SERVICES AND THREADS STRICTLY AFTER SYNCHRONOUS LOAD
    t_audio = threading.Thread(target=audio_stream_worker, daemon=True)
    t_gen = threading.Thread(target=tts_subprocess_worker, daemon=True)
    t_play = threading.Thread(target=tts_playback_worker, daemon=True)
    
    t_audio.start()
    t_gen.start()
    t_play.start()

    stdscr.nodelay(True)
    stdscr.timeout(100)
    
    transcriber_started = False
    config_check_counter = 0

    try:
        while should_run:
            config_check_counter += 1
            if config_check_counter >= 10:
                check_config_reload()
                config_check_counter = 0

            if soundboard_visible:
                t_h = h // 2
                transcript_win.resize(t_h, w - half)
                transcript_win.mvwin(0, half)
                update_transcript(transcript_win)
                
                if soundboard_win is None:
                    soundboard_win = curses.newwin(h - t_h - 1, w - half, t_h, half)
                update_soundboard(soundboard_win)
            else:
                transcript_win.resize(h - 2, w - half)
                transcript_win.mvwin(0, half)
                update_transcript(transcript_win)
                
            update_settings(settings_win)
            update_logs(log_win)
            update_command(cmd_win)
            
            key = stdscr.getch()
            
            if key != -1:
                if soundboard_visible:
                    if key == curses.KEY_UP: sound_selected = max(0, sound_selected - 1)
                    elif key == curses.KEY_DOWN: sound_selected = min(len(sound_files) - 1, sound_selected + 1)
                    elif key in (10, 13):
                        if sound_files: play_sound(sound_files[sound_selected])
                    elif key == ord('s'): soundboard_visible = False
                    elif key == curses.KEY_MOUSE and soundboard_win:
                        selected_idx = handle_mouse_event(soundboard_win)
                        if selected_idx is not None:
                            sound_selected = selected_idx
                            play_sound(sound_files[sound_selected])
                    elif key == ord('e'):
                        if sound_files: edit_sound_emoji(sound_selected)
                else:
                    if key == curses.KEY_UP: selected_index = max(0, selected_index - 1)
                    elif key == curses.KEY_DOWN: selected_index = min(len(setting_keys) - 1, selected_index + 1)
                    elif key == ord('\n') or key == ord(' '): edit_setting(settings_win)
                    elif key == curses.KEY_PPAGE: log_scroll = min(MAX_LOG_LINES, log_scroll + 1)
                    elif key == curses.KEY_NPAGE: log_scroll = max(0, log_scroll - 1)
                    elif key == ord('s'):
                         try: sound_files = sorted([f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.wav', '.mp3'))])
                         except: pass
                         soundboard_visible = True
                    elif key == ord('q'): should_run = False
            
            if model_ready.is_set() and not transcriber_started:
                threading.Thread(target=transcribe_worker, daemon=True).start()
                transcriber_started = True

    except Exception as e:
        with open("crash.log", "w") as f:
            f.write(str(e))
    finally:
        final_cleanup()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nExiting...")