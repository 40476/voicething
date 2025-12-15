#!/usr/bin/python3
import faulthandler
faulthandler.enable()
import curses, time, librosa, queue, threading, numpy as np, textwrap, subprocess, signal, wave, tempfile, whisper, pyaudio, torch, os, json, resampy
from datetime import datetime
from dbus_next.aio import MessageBus
from dbus_next.service import ServiceInterface, method
from pathlib import Path

# =============== CONFIG PATHS ===============
CONFIG_DIR = Path.home() / ".local/share/voicething"
SOUNDS_DIR = CONFIG_DIR / "sounds"
CONFIG_FILE = CONFIG_DIR / "voicething.json"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

# =============== DEFAULT SETTINGS ===============
default_settings = {
    "whisper_model": "small",
    "whisper_language": "en",
    "voice_model": "en_US/hifi-tts_low",
    "audio_device": None,
    "tts_enabled": True,
    "prosody_enabled": True,
    "sound_emojis": {},
    "ptt_active": False,
}

# Load or create config
if CONFIG_FILE.exists():
    try:
        settings = json.loads(CONFIG_FILE.read_text())
        # Merge with defaults for missing keys
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
should_run = True
torch.backends.cudnn.enabled = False

# =============== SOUNDBOARD STATE ===============
# Add to soundboard state initialization
soundboard_visible = True
sound_files = []
sound_selected = 0
sound_scroll = 0
last_play_time = 0

# =============== STATE ===============
transcript_log = []
log_lines = []
log_scroll = 0
MAX_LOG_LINES = 100
audio_q = queue.Queue()
log_q = queue.Queue()
child_procs = []
model_ready = threading.Event()
model = None

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


# ================ SOUND PLAYBACK ================
def play_sound(filename):
    global last_play_time
    current_time = time.time()
    
    # 1-second cooldown check
    if current_time - last_play_time < 1.0:
        return
        
    last_play_time = current_time

    """Play sound through both TTS voice and system audio"""
    filepath = SOUNDS_DIR / filename
    sound_name = os.path.splitext(filename)[0].replace("_", " ")
    
    try:
        subprocess.Popen([
            "paplay", "--device=TTS_voice", "--volume=80000", filepath
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except Exception as e:
        log(f"[sound] Error: {e}")


# =============== HELPERS ===============
def handle_mouse_event(win):
    try:
        # Get mouse event details
        _, mx, my, _, btn_state = curses.getmouse()

        # Only allow left mouse button presses
        if not (btn_state & curses.BUTTON1_PRESSED):
            return None

        # Get window boundaries
        win_start_y, win_start_x = win.getbegyx()
        win_height, win_width = win.getmaxyx()

        # Check if click is INSIDE the window
        if not (win_start_x <= mx < win_start_x + win_width and
                win_start_y <= my < win_start_y + win_height):
            return None

        # Convert to window coordinates
        rel_y = my - win_start_y
        rel_x = mx - win_start_x

        # Only process clicks in CONTENT AREA (inside borders)
        if not (1 <= rel_x < win_width - 1 and 1 <= rel_y < win_height - 1):
            return None

        # Calculate clicked button row (each takes 3 lines)
        button_row = (rel_y - 1) // 3  # Remove top border offset

        # Calculate sound index based on scroll
        sound_index = sound_scroll + button_row

        # Validate index range
        if 0 <= sound_index < len(sound_files):
            return sound_index
        
        return None
    except Exception as e:
        log(f"[Mouse] Error: {e}")
        return None

def save_config():
    CONFIG_FILE.write_text(json.dumps(settings, indent=2))

# =============== LOGGING ===============
def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_q.put(f"[{timestamp}] {msg}")

# =============== AUDIO ===============
def play_tone(frequency=440, duration=0.05, sink="TTS_voice"):
    try:
        cmd = f"sox -n -r 44100 -c 2 -b 16 -t raw - synth {duration} sine {frequency}"
        if sink:
            cmd += f" | pacat --client-name=TonePlayer --device={sink}"
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"[tone] Error: {e}")

def edit_sound_emoji(idx):
    filename = sound_files[idx]
    curses.echo()
    
    # Create editor window
    h, w = curses.LINES, curses.COLS
    edit_win = curses.newwin(3, 40, h//2-1, w//2-20)
    edit_win.border()
    edit_win.addstr(0, 2, " EDIT EMOJI ")
    edit_win.addstr(1, 2, f"{filename}: ")
    
    # Get new emoji
    edit_win.refresh()
    emoji = edit_win.getstr(1, len(filename) + 4, 3).decode().strip()
    curses.noecho()
    
    if emoji:
        settings["sound_emojis"][filename] = emoji
        save_config()
        log(f"Updated emoji for {filename} → {emoji}")

# =============== PTT VIA DBUS ===============
class PTTInterface(ServiceInterface):
    def __init__(self):
        super().__init__('com.speak.PTT')

    @method()
    def Toggle(self) -> "s": # this fine dont touch it it works idk why or how
        settings["ptt_active"] = not settings.get("ptt_active", False)
        volume_path = os.path.expanduser("~/.local/state/speakvolumerc")
        if settings["ptt_active"]:
            subprocess.call(f"echo -n \"$(awk -F'[][]' '/Left:/ {{ print $2 }}' <(amixer sget Master))\" > {volume_path}", shell=True)
            subprocess.call("amixer set Master 20%", shell=True)
            play_tone(600, 0.3)  # Start tone
        else:
            subprocess.call(f"amixer set Master $(cat {volume_path})", shell=True)
            play_tone(440, 0.2)  # Processing tone
        log(f"[ptt] {'Activated' if settings['ptt_active'] else 'Deactivated'}")
        return "PTT toggled"

def start_ptt_listener():
    import asyncio

    async def runner():
        bus = await MessageBus().connect()
        interface = PTTInterface()
        bus.export('/com/speak/PTT', interface)
        await bus.request_name('com.speak.PTT')
        log("[DBus] PTT service running...")
        await asyncio.get_event_loop().create_future()

    threading.Thread(target=lambda: asyncio.run(runner()), daemon=True).start()

def list_audio_devices():
    pa = pyaudio.PyAudio()
    input_devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            input_devices.append((i, info["name"]))
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
        max_offset = max(0, len(devices) - visible_height)

        if selected_index < scroll_offset:
            scroll_offset = selected_index
        elif selected_index >= scroll_offset + visible_height:
            scroll_offset = selected_index - visible_height + 1

        for i in range(visible_height):
            idx = scroll_offset + i
            if idx >= len(devices):
                break
            line = f"{idx}: {devices[idx]}"
            color = curses.color_pair(1 if idx == selected_index else 2)
            win.attrset(color)
            win.addstr(i + 1, 2, line[:w - 4])

        win.refresh()

        key = win.getch()
        if key == curses.KEY_UP and selected_index > 0:
            selected_index -= 1
        elif key == curses.KEY_DOWN and selected_index < len(devices) - 1:
            selected_index += 1
        elif key in (10, 13):
            return selected_index
        elif key == 27:  # ESC
            return None

def audio_stream_worker():
    pa = pyaudio.PyAudio()
    stream = None
    previous_device = None

    while True:
        try:
            device_id = settings.get("audio_device")

            if device_id is None:
                time.sleep(0.5)
                continue

            if previous_device != device_id:
                previous_device = device_id
                if stream:
                    log("[audio] Stopping stream")
                    stream.stop_stream()
                    stream.close()
                    stream = None

            info = pa.get_device_info_by_index(device_id)
            if info["maxInputChannels"] == 0:
                log(f"[audio] Device {device_id} has no input channels — retrying")
                time.sleep(0.5)
                continue

            if stream is None:
                log(f"[audio] Opening stream on device {device_id}")
                stream = pa.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device_id
                )
                stream.start_stream()
                log(f"[audio] Stream started on {info['name']}")

            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_q.put(data)
            time.sleep(0.01)

        except Exception as e:
            log(f"[audio] Stream error: {e}")
            if stream:
                stream.stop_stream()
                stream.close()
                stream = None
            time.sleep(1)

# =============== SPEAK ===============
def map_range(val, in_min, in_max, out_min, out_max):
    """
    Linearly map a value from one range to another.
    Clamps to out_min/out_max if val is outside in_min/in_max.
    """
    if in_max == in_min:
        return out_min  # avoid division by zero
    # Normalize into 0–1
    ratio = (val - in_min) / (in_max - in_min)
    # Clamp ratio
    ratio = max(0.0, min(1.0, ratio))
    # Scale into output range
    return out_min + ratio * (out_max - out_min)

def extract_prosody(audio_np, sr=16000):
    try:
        pitches, magnitudes = librosa.piptrack(y=audio_np, sr=sr)
        pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches > 0) else 0
        energy = float(np.mean(librosa.feature.rms(y=audio_np)))
        tempo, _ = librosa.beat.beat_track(y=audio_np, sr=sr)
        return {"pitch": pitch, "energy": energy, "tempo": tempo}
    except Exception as e:
        log(f"[prosody] Error extracting: {e}")
        return {"pitch": 50, "energy": 0.05, "tempo": 120}

def map_prosody_to_mimic3(prosody):
    if settings.get("prosody_enabled", True) is False:
        return 1.0, 1.0, 1.0  # Default params if prosody disabled
    # Expect prosody dict with keys: pitch (Hz), energy, tempo (BPM)

    raw_pitch = prosody.get("pitch", 150)
    raw_energy = prosody.get("energy", 0.05)
    raw_tempo = prosody.get("tempo", 120)

    # Map pitch into noise_w (prosody randomness)
    noise_w = map_range(raw_pitch, 80, 400, 0.5, 1.2)

    # Map energy into noise_scale (expressiveness)
    noise_scale = map_range(raw_energy, 0.01, 0.1, 0.5, 1.0)

    # Map tempo into length_scale (speaking rate)
    length_scale = map_range(raw_tempo, 60, 200, 0.8, 1.2)

    return length_scale, noise_scale, noise_w

def speak(text, prosody=None):
    if settings.get("tts_enabled", True):
        try:
            # Default prosody-driven params
            length_scale, noise_scale, noise_w = 1.0, 1.0, 1.0
            if prosody :
                length_scale, noise_scale, noise_w = map_prosody_to_mimic3(prosody)
                # Invert length_scale for Mimic3
                length_scale, noise_scale, noise_w = str(2-(float(length_scale)**2)), str(noise_scale), str(noise_w)
                log(f"[prosody] length_scale={float(length_scale)}, "
                        f"noise_scale={noise_scale}, noise_w={noise_w}")
            
            # Build CLI command
            cmd = [
                "python3", "-m", "mimic3_tts",
                "--voice", settings["voice_model"],
                "--noise-scale", noise_scale,
                "--length-scale", length_scale,
                "--noise-w", str(0),
                text
            ]

            log(f"[TTS] Built command: {' '.join(cmd)}")
            # Pipe directly into pacat
            proc = subprocess.Popen(
                cmd + ["--stdout"],  # dump WAV to stdout
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            pacat = subprocess.Popen(
                ["pacat", "--client-name=TonePlayer", "--device=TTS_voice",
                  "--rate=22050", "--channels=1", "--format=s16le"],
                stdin=proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            pacat.communicate()
            proc.wait()

        except Exception as e:
            log(f"[speak error] {e}")

# =============== WHISPER ===============
def load_model_async():
    global model
    try:
        log(f"[model] Loading Whisper: {settings['whisper_model']}")
        model = whisper.load_model(settings["whisper_model"])
        model_ready.set()
        log("[model] Ready.")
    except Exception as e:
        log(f"[model] Load failed: {e}")

# =============== TRANSCRIBE ===============
def transcribe_worker(model, transcript_win=None):
    torch.backends.cudnn.enabled = False
    frames = []
    was_recording = False
    last_beep_time = 0

    log("[transcribe] Ready")

    while True:
        try:
            data = audio_q.get()
            ptt_active = settings.get("ptt_active", False)

            if ptt_active:
                if time.time() - last_beep_time > 1:
                    play_tone(700, 0.1)
                    last_beep_time = time.time()
                if not was_recording:
                    log("[transcribe] PTT activated — starting buffer")
                    was_recording = True
                    time.sleep(0.5)
                frames.append(np.frombuffer(data, dtype=np.int16))

            elif was_recording:
                was_recording = False
                log("[transcribe] PTT released — processing buffer")

                with audio_q.mutex:
                    audio_q.queue.clear()

                if not frames:
                    log("[transcribe] Buffer empty — skipping")
                    continue

                audio_np = np.hstack(frames).astype(np.float32)
                frames = []

                if settings.get("noise_cancel", False):
                    threshold = 0.02
                    audio_np = np.where(np.abs(audio_np) < threshold, 0, audio_np)

                audio_np /= 32768.0

                input_rate = settings.get("sample_rate", 16000)
                if input_rate != 16000:
                    audio_np = resampy.resample(audio_np, input_rate, 16000)

                if np.max(np.abs(audio_np)) < 0.01:
                    log("[transcribe] Quiet buffer — skipping")
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    with wave.open(f.name, "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(RATE)
                        wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())

                    result = model.transcribe(f.name, language=settings.get("whisper_language", "en"), fp16=True)
                    text = result.get("text", "").strip()

                    if text:
                        timestamp = datetime.now().strftime("[%H:%M:%S]")
                        full_text = f"{timestamp} {text}"
                        log(f"[transcribe] {full_text}")
                        play_tone(1040, 0.2)

                        if transcript_log is not None:
                          timestamp = datetime.now().strftime("[%H:%M:%S]")
                          full_text = f"{timestamp} {text}"
                          transcript_log.append(full_text)

                        prosody = extract_prosody(audio_np)
                        speak(text, prosody)

        except Exception as e:
            log(f"[transcribe] Error: {e}")

# =============== UI ===============
def update_settings(win):
    """
    Dynamically update the settings panel based on the current settings.
    """
    win.clear()
    win.box()
    win.addstr(0, 2, " SETTINGS ")
    for i, key in enumerate(setting_keys):
        val = str(settings[key])
        val_type = type(default_settings[key]).__name__
        line = f"{key:<14} ({val_type}): {val}"
        if i == selected_index:
            win.attron(curses.A_REVERSE)
            win.addstr(i + 2, 2, line.ljust(40))
            win.attroff(curses.A_REVERSE)
        else:
            win.addstr(i + 2, 2, line.ljust(40))
    win.refresh()

def update_logs(win):
    global log_scroll
    h, w = win.getmaxyx()
    win.clear()
    win.box()
    win.addstr(0, 2, " LOGS ")
    
    while not log_q.empty():
        log_lines.append(log_q.get())
        if len(log_lines) > MAX_LOG_LINES: 
            log_lines.pop(0)
    
    visible_lines = []
    if log_scroll > 0:
        start_idx = min(log_scroll, len(log_lines) - 1)
        visible_lines = log_lines[-start_idx:]
    else:
        visible_lines = log_lines[-h + 2:]
    
    row = 1
    for line in visible_lines:
        wrapped = textwrap.wrap(line, w - 4)
        for wrapped_line in wrapped:
            if row >= h - 1: 
                break
            win.attrset(curses.color_pair(1))
            win.addstr(row, 2, wrapped_line)
            row += 1
    win.refresh()

def update_transcript(win):
    h, w = win.getmaxyx()
    win.clear()
    win.box()
    win.addstr(0, 2, " TRANSCRIPT ")

    wrapped_lines = []
    for line in transcript_log[-MAX_LOG_LINES:]:
        wrapped = textwrap.wrap(line, width=w - 4)
        wrapped_lines.extend(wrapped)

    visible_count = h - 2
    visible_lines = wrapped_lines[-visible_count:] if len(wrapped_lines) > visible_count else wrapped_lines

    for idx, line in enumerate(visible_lines):
        if idx < h - 2:
            color = curses.color_pair(1 if idx % 2 == 0 else 2)
            win.attrset(color)
            win.addstr(idx + 1, 2, line)

    win.refresh()

# ================ SOUNDBOARD UI ================
def update_soundboard(win):
    global sound_files, sound_scroll, sound_selected
    try:
        h, w = win.getmaxyx()
        
        # Early return if window too small
        if h < 4 or w < 10:
            win.addstr(1, 2, "Window too small")
            win.refresh()
            return
            
        win.clear()
        win.border()
        win.addstr(0, 2, " SOUNDBOARD [S] (▲/▼ scroll) ")

        num_items = len(sound_files)
        btn_h = 3  # Fixed button height
        visible_btns = max(1, (h - 2) // btn_h)
        max_scroll = max(0, num_items - visible_btns)
        
        # ================== CRITICAL SCROLL FIX ==================
        sound_selected = max(0, min(sound_selected, num_items - 1 if num_items > 0 else 0))
        
        # Keep selection visible
        if num_items > 0:
            if sound_selected < sound_scroll:
                sound_scroll = max(0, sound_selected)
            elif sound_selected >= sound_scroll + visible_btns:
                sound_scroll = min(max_scroll, sound_selected - visible_btns + 1)
        sound_scroll = max(0, min(sound_scroll, max_scroll))

        # Draw header/no sounds message
        if num_items == 0:
            win.addstr(1, 2, " ➕ Add sounds to ~/.local/share/voicething/sounds")
            win.refresh()
            return
            
        # Draw buttons
        for i in range(visible_btns):
            idx = sound_scroll + i
            if idx >= num_items:
                break
                
            filename = sound_files[idx]
            emoji = settings["sound_emojis"].get(filename, "🔊")
            name = os.path.splitext(filename)[0][:30]
            
            # Calculate button position
            btn_top = i * btn_h + 1
            
            # Skip if bottom exceeds window
            if btn_top + btn_h - 1 >= h:
                break
                
            # Selection highlighting
            is_selected = idx == sound_selected
            color_id = 2 if is_selected else 1
            win.attrset(curses.color_pair(color_id))
            
            # Draw button box
            win.addstr(btn_top, 2, "┌" + "─"*(w-6) + "┐")
            text_line = f"│ {emoji} {name.center(w-6)[2:]} "[:w-2]
            win.addstr(btn_top+1, 2, text_line + "│".rjust(w-len(text_line)-2))
            win.addstr(btn_top+2, 2, "└" + "─"*(w-6) + "┘")

        win.refresh()
    except Exception as e:
        log(f"[SOUNDBOARD] {str(e)[:30]}")

def safe_addstr(win, y, x, text):
    """Safely write text to window coordinates"""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x < 0 or x >= max_x - 2:
        return
    text = text[:max_x - x - 1]
    try:
        win.addstr(y, x, text)
    except:
        pass  # Ignore edge write errors

def update_command(win):
    win.clear()
    win.box()
    win.addstr(1, 2, " LAUNCH COMMAND ")
    cmd = f"python3 main.py --model {settings['whisper_model']}"
    win.addstr(2, 2, cmd[:win.getmaxyx()[1] - 4])
    win.refresh()

def validate_and_cast(value, expected_type):
    """
    Validate and cast the input value to the expected type.
    Raise ValueError if the value cannot be cast.
    """
    try:
        if expected_type == bool:
            # Special handling for booleans
            if value.lower() in ("true", "1", "yes", "on"):
                return True
            elif value.lower() in ("false", "0", "no", "off"):
                return False
            else:
                raise ValueError("Invalid boolean value")
        return expected_type(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid value: {value}. Expected type: {expected_type.__name__}")

def edit_setting(win):
    """
    Edit the selected setting, ensuring the value matches the expected type.
    """
    global selected_index
    key = setting_keys[selected_index]
    expected_type = type(default_settings[key])

    if key == "audio_device":
        devices = list_audio_devices()
        dev_names = [f"{idx}: {name}" for idx, name in devices]
        try:
            sel_index = device_selector(win, dev_names)
            if sel_index is not None:
                settings["audio_device"] = devices[sel_index][0]
                log(f"[audio] Selected device: {devices[sel_index][1]}")
                save_config()
        except Exception as e:
            log(f"[audio] Invalid selection: {e}")

    else:
        curses.echo()
        prompt_y = len(setting_keys) + 3
        win.addstr(prompt_y, 2, f"{key} ({expected_type.__name__}): ")
        val = win.getstr(prompt_y, len(key) + len(expected_type.__name__) + 5, 20).decode()
        curses.noecho()
        try:
            settings[key] = validate_and_cast(val, expected_type)
            log(f"[settings] {key} → {settings[key]}")
            save_config()
        except ValueError as e:
            log(f"[settings] Error: {e}")

def handle_sigint(sig, frame):
    global should_run
    should_run = False
    log("SIGINT received — shutting down.")

def cleanup_children():
    if child_procs:
        log("Cleaning up child processes...")
        for proc in child_procs:
            try: 
                proc.terminate()
                proc.wait(timeout=0.5)
            except: 
                pass
    log("All child processes cleaned up.")

def main(stdscr):
    global selected_index, log_scroll, should_run
    global soundboard_visible, sound_selected, sound_files
    
    def final_cleanup():
        cleanup_children()
        save_config()
        log("Config saved")
        os._exit(0)  # Force clean exit
    
    signal.signal(signal.SIGINT, lambda s, f: final_cleanup())
    
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    
    h, w = stdscr.getmaxyx()
    half = w // 2
    
    # Window initialization
    settings_win = curses.newwin(12, half, 0, 0)
    log_win = curses.newwin(h - 17, half, 12, 0)
    cmd_win = curses.newwin(5, half, h - 5, 0)
    transcript_win = curses.newwin(h - 2, w - half, 0, half)
    soundboard_win = None
    
    sound_files = sorted([f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.wav', '.mp3'))])
    
    start_ptt_listener()
    threading.Thread(target=load_model_async, daemon=True).start()
    threading.Thread(target=audio_stream_worker, daemon=True).start()
    log("[init] Threads started.")

    stdscr.nodelay(True)
    stdscr.timeout(100)
    
    try:
        while should_run:
            if soundboard_visible:
                # Update window sizes when soundboard is visible
                transcript_height = h // 2
                transcript_win.resize(transcript_height, w - half)
                transcript_win.mvwin(0, half)
                update_transcript(transcript_win)
                
                if soundboard_win is None:
                    soundboard_win = curses.newwin(h - transcript_height - 1, w - half, transcript_height, half)
                update_soundboard(soundboard_win)
            else:
                transcript_win.resize(h - 2, w - half)
                transcript_win.mvwin(0, half)
                update_transcript(transcript_win)
                
            # Update other UI components
            update_settings(settings_win)
            update_logs(log_win)
            update_command(cmd_win)
            
            time.sleep(0.05)
            key = stdscr.getch()
            
            if key != -1:
                if soundboard_visible:
                    # Soundboard navigation
                    if key == curses.KEY_UP:
                        sound_selected = max(0, sound_selected - 1)
                    elif key == curses.KEY_DOWN:
                        sound_selected = min(len(sound_files) - 1, sound_selected + 1)
                    elif key in (10, 13):  # Enter
                        if sound_files:
                            play_sound(sound_files[sound_selected])
                    elif key == ord('s'):
                        soundboard_visible = False
                    elif key == curses.KEY_MOUSE and soundboard_visible and soundboard_win:
                        selected_idx = handle_mouse_event(soundboard_win)  # Removed extra args
                        if selected_idx is not None:
                            sound_selected = selected_idx
                            play_sound(sound_files[sound_selected])
                    elif key == ord('e') and soundboard_visible:
                        if sound_files:
                            edit_sound_emoji(sound_selected)
                else:
                    # Main navigation
                    if key == curses.KEY_UP:
                        selected_index = max(0, selected_index - 1)
                    elif key == curses.KEY_DOWN:
                        selected_index = min(len(setting_keys) - 1, selected_index + 1)
                    elif key == ord('\n') or key == ord(' '):
                        edit_setting(settings_win)
                    elif key == curses.KEY_PPAGE:
                        log_scroll = min(MAX_LOG_LINES, log_scroll + 1)
                    elif key == curses.KEY_NPAGE:
                        log_scroll = max(0, log_scroll - 1)
                    elif key == ord('t'):
                        with open("transcript.txt", "w") as f:
                            f.write("\n".join(transcript_log))
                        log("Transcript saved.")
                    elif key == ord('s'):
                        # Add this to your 'S' key handler
                        sound_files = sorted([f for f in os.listdir(SOUNDS_DIR) if f.endswith(('.wav', '.mp3'))])
                        soundboard_visible = True
                        sound_selected = 0
                    elif key == ord('q'):
                        should_run = False
                        log("Quitting...")
            
            # Start transcribe worker when ready
            if model_ready.is_set() and not any(t.name == "transcriber" for t in threading.enumerate()):
                threading.Thread(target=transcribe_worker,
                                args=(model, transcript_win),
                                daemon=True,
                                name="transcriber").start()
    finally:
        final_cleanup()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nExiting...")