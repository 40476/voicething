import subprocess
def play_tone(frequency=440, duration=0.05, sink="TTS_voice"):
    try:
        cmd = f"sox -n -r 44100 -c 2 -b 16 -t raw - synth {duration} sine {frequency}"
        if sink:
            cmd += f" | pacat --client-name=TonePlayer --device={sink}"
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[tone] Error: {e}")
play_tone(600, 0.1)  # Start tone