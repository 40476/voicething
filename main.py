#!/usr/bin/python3
# import faulthandler
# faulthandler.enable()
import curses, time, queue, threading, numpy as np, textwrap, subprocess, signal, wave, tempfile, whisper, pyaudio, torch, os
from datetime import datetime
from dbus_next.aio import MessageBus
from dbus_next.service import ServiceInterface, method

# =============== SETTINGS ===============
settings = {
    "model": "small",
    "language": "en",
    "audio_device":14,
    "speak": True,
    
}
setting_keys = list(settings.keys())
selected_index = 0
should_run = True
torch.backends.cudnn.enabled = False

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

# =============== LOGGING ===============
def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_q.put(f"[{timestamp}] {msg}")

# =============== AUDIO ===============
class PTTInterface(ServiceInterface):
    def __init__(self):
        super().__init__('com.speak.PTT')

    @method()
    def Toggle(self) -> 's':
        settings["ptt_active"] = not settings.get("ptt_active", False)
        volume_path = os.path.expanduser("~/.local/state/speakvolumerc")
        if settings["ptt_active"]:
            
            subprocess.call(f"echo -n \"$(awk -F'[][]' '/Left:/ {{ print $2 }}' <(amixer sget Master))\" > {volume_path}", shell=True)
            subprocess.call("amixer set Master 20%", shell=True)
        else:
            subprocess.call(f"amixer set Master $(cat {volume_path})",shell=True)
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
        visible_height = h - 2  # room for borders
        max_offset = max(0, len(devices) - visible_height)

        # Scroll logic
        if selected_index < scroll_offset:
            scroll_offset = selected_index
        elif selected_index >= scroll_offset + visible_height:
            scroll_offset = selected_index - visible_height + 1

        # Show visible chunk
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
        elif key in (10, 13):  # Enter
            return selected_index + 1

def audio_stream_worker():
    pa = pyaudio.PyAudio()
    stream = None
    previous_device = None

    while True:
        try:
            device_id = settings.get("audio_device")

            # Wait until a device is selected
            if device_id is None:
                time.sleep(0.5)
                continue

            # If the selected device has changed, reset stream
            if previous_device != device_id:
                previous_device = device_id
                if stream:
                    log("[audio] Stopping stream")
                    stream.stop_stream()
                    stream.close()
                    stream = None

            # Verify device is input-capable
            info = pa.get_device_info_by_index(device_id)
            if info["maxInputChannels"] == 0:
                log(f"[audio] Device {device_id} has no input channels — retrying")
                time.sleep(0.5)
                continue

            # Lazy open stream if not already active
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

            # Add your stream reading logic here (e.g., stream.read() → audio_q.put())
            # Example placeholder:
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
def speak(text):
    try:
        proc = subprocess.Popen(["espeak", "-d", "TTS_voice", text],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        child_procs.append(proc)
    except Exception as e:
        log(f"[speak error] {e}")

# =============== WHISPER ===============
def load_model_async():
    global model
    try:
        log(f"[model] Loading Whisper: {settings['model']}")
        model = whisper.load_model(settings["model"])
        model_ready.set()
        log("[model] Ready.")
    except Exception as e:
        log(f"[model] Load failed: {e}")

# =============== TRANSCRIBE ===============
def transcribe_worker(model, transcript_win=None):
    torch.backends.cudnn.enabled = False
    frames = []
    was_recording = False

    log("[transcribe] Ready")

    while True:
        try:
            data = audio_q.get()
            ptt_active = settings.get("ptt_active", False)

            if ptt_active:
                if not was_recording:
                    log("[transcribe] PTT activated — starting buffer")
                    was_recording = True
                    time.sleep(0.5)  # Optional delay
                frames.append(np.frombuffer(data, dtype=np.int16))

            elif was_recording:
                # PTT released — process buffer
                was_recording = False
                log("[transcribe] PTT released — processing buffer")

                # Flush audio queue to prevent overlap
                with audio_q.mutex:
                    audio_q.queue.clear()

                if not frames:
                    log("[transcribe] Buffer empty — skipping")
                    continue

                # Convert and clean buffer
                audio_np = np.hstack(frames).astype(np.float32)
                frames = []

                if settings.get("noise_cancel", False):
                    threshold = 0.02
                    audio_np = np.where(np.abs(audio_np) < threshold, 0, audio_np)

                audio_np /= 32768.0  # Normalize

                if np.max(np.abs(audio_np)) < 0.01:
                    log("[transcribe] Quiet buffer — skipping")
                    continue

                # Save to temp WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    with wave.open(f.name, "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(RATE)
                        wf.writeframes((audio_np * 32768).astype(np.int16).tobytes())

                    result = model.transcribe(f.name, language=settings.get("language", "en"), fp16=True)
                    text = result.get("text", "").strip()

                    if text:
                        timestamp = datetime.now().strftime("[%H:%M:%S]")
                        full_text = f"{timestamp} {text}"
                        log(f"[transcribe] {full_text}")

                        # Write to transcript log with wrapping
                        if transcript_log is not None:
                          timestamp = datetime.now().strftime("[%H:%M:%S]")
                          full_text = f"{timestamp} {text}"
                          transcript_log.append(full_text)


                        # Speak the result if enabled
                        speak(text)

        except Exception as e:
            log(f"[transcribe] Error: {e}")

# =============== UI ===============
def update_settings(win):
    win.clear(); win.box(); win.addstr(0, 2, " SETTINGS ")
    for i, key in enumerate(setting_keys):
        val = str(settings[key])
        line = f"{key:<14}: {val}"
        if i == selected_index:
            win.attron(curses.A_REVERSE)
            win.addstr(i + 2, 2, line.ljust(40))
            win.attroff(curses.A_REVERSE)
        else:
            win.addstr(i + 2, 2, line.ljust(40))
    win.refresh()

def update_logs(win):
    global log_scroll
    h, w = win.getmaxyx(); win.clear(); win.box(); win.addstr(0, 2, " LOGS ")
    while not log_q.empty():
        log_lines.append(log_q.get())
        if len(log_lines) > MAX_LOG_LINES: log_lines.pop(0)
    lines = log_lines[-(h - 2 + log_scroll):-log_scroll] if log_scroll else log_lines[-(h - 2):]
    row = 1
    for idx, line in enumerate(lines):
        for wrapped in textwrap.wrap(line, w - 4):
            if row >= h - 1: break
            win.attrset(curses.color_pair(1 if row % 2 == 0 else 2))
            win.addstr(row, 2, wrapped); row += 1
    win.refresh()

def update_transcript(win):
    h, w = win.getmaxyx()
    win.clear()
    win.box()
    win.addstr(0, 2, " TRANSCRIPT ")

    wrapped_lines = []
    for line in transcript_log:
        wrapped = textwrap.wrap(line, width=w - 4)
        wrapped_lines.extend(wrapped)

    visible_lines = wrapped_lines[-(h - 2):]
    for idx, line in enumerate(visible_lines):
        color = curses.color_pair(1 if idx % 2 == 0 else 2)
        win.attrset(color)
        win.addstr(idx + 1, 2, line)

    win.refresh()


def update_command(win):
    win.clear(); win.box(); win.addstr(1, 2, " LAUNCH COMMAND ")
    cmd = f"python3 main.py --model {settings['model']}"
    win.addstr(2, 2, cmd[:win.getmaxyx()[1] - 4]); win.refresh()

def edit_setting(win):
    global selected_index
    key = setting_keys[selected_index]

    if key == "speak":
        settings["speak"] = not settings["speak"]
        log(f"Speech {'enabled' if settings['speak'] else 'disabled'}")

    elif key == "audio_device":
        devices = list_audio_devices()
        

        try:
            sel_index = int(device_selector(win,devices)) - 1
            settings["audio_device"] = devices[sel_index][0]  # PyAudio index
            log(f"[audio] Selected device: {devices[sel_index][1]}")
        except Exception as e:
            log(f"[audio] Invalid selection: {e}")

    else:
        curses.echo()
        prompt_y = len(setting_keys) + 3
        win.addstr(prompt_y, 2, f"{key}: ")
        val = win.getstr(prompt_y, len(key) + 4, 20).decode()
        curses.noecho()
        try:
            settings[key] = float(val) if key == "buffer_seconds" else val
            log(f"[settings] {key} → {settings[key]}")
        except Exception as e:
            log(f"[settings] Error: {e}")


def handle_sigint(sig, frame):
    global should_run
    should_run = False
    log("SIGINT received — shutting down.")

def cleanup_children():
    if child_procs:
        log("Cleaning up child processes...")
        for proc in child_procs:
            try: proc.terminate()
            except: pass
    log("All child processes cleaned up.")

def main(stdscr):
    global selected_index, log_scroll, should_run
    curses.curs_set(0); curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    h, w = stdscr.getmaxyx(); half = w // 2
    settings_win   = curses.newwin(len(setting_keys) + 6, half, 0, 0)
    log_win        = curses.newwin(h - settings_win.getmaxyx()[0] - 5, half, settings_win.getmaxyx()[0], 0)
    cmd_win        = curses.newwin(5, half, h - 5, 0)
    transcript_win = curses.newwin(h, w - half, 0, half)
    
    start_ptt_listener()
    threading.Thread(target=load_model_async, daemon=True).start()
    threading.Thread(target=audio_stream_worker, daemon=True).start()
    log("[init] Threads started.")

    stdscr.nodelay(True); stdscr.timeout(100)
    try:
        while should_run:
            update_settings(settings_win)
            update_transcript(transcript_win)
            update_logs(log_win)
            update_command(cmd_win)
            time.sleep(0.05)
            key = stdscr.getch()
            if key != -1:
                if key == curses.KEY_UP: selected_index = max(0, selected_index - 1)
                elif key == curses.KEY_DOWN: selected_index = min(len(setting_keys) - 1, selected_index + 1)
                elif key == ord('\n') or key == ord(' '): edit_setting(settings_win)
                elif key == curses.KEY_PPAGE: log_scroll = min(MAX_LOG_LINES, log_scroll + 1)
                elif key == curses.KEY_NPAGE: log_scroll = max(0, log_scroll - 1)
                elif key == ord('s'):
                    with open("transcript.txt", "w") as f:
                        f.write("\n".join(transcript_log))
                    log("Transcript saved.")
                elif key == ord('q'):
                    should_run = False
                    log("Quitting...")
                    cleanup_children()
                    hujgkvsdhjbgkhjdfhjkdfshjjhklzzgjhklgfzjhk
                    break

            if model_ready.is_set() and not any(t.name == "transcriber" for t in threading.enumerate()):
                threading.Thread(target=transcribe_worker,
                                  args=(model, transcript_win),
                                  daemon=True,
                                  name="transcriber").start()
    finally:
        log("Shutting down UI...")
        cleanup_children()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    curses.wrapper(main)
