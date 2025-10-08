import pyaudio

print("=== Input Devices ===")
pa = pyaudio.PyAudio()
input_devices = []
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        for key, value in info.items():
            print(f"  {key}: {value}")
