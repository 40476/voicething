#!/bin/bash

sleep 2

# Create a virtual mic source
pactl load-module module-null-sink media.class=Audio/Source/Virtual sink_name=VirtualMic channel_map=front-left,front-right

# Link TTS_voice output to VirtualMic input (loopback for mic simulation)
pw-link TTS_voice:monitor_FL VirtualMic:input_FL
pw-link TTS_voice:monitor_FR VirtualMic:input_FR

# Create a null sink for TTS output
pactl load-module module-null-sink sink_name=TTS_voice sink_properties=device.description=TTS_voice

# Create loopback from TTS_voice to system default audio
DEFAULT_SINK=$(pactl info | grep "Default Sink" | awk '{print $3}')
pactl load-module module-loopback source=TTS_voice.monitor sink=$DEFAULT_SINK

# Create a null sink for TTS output again (sometimes doesnt  fails silently)
# pactl load-module module-null-sink sink_name=TTS_voice sink_properties=device.description=TTS_voice

# Notify success
notify-send -a "Audio Split" "TTS_voice routed to system + virtual mic"
