import pyaudio
import wave

# Audio Configuration (The 16km/20L Standard)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recitation.wav"

audio = pyaudio.PyAudio()

print("--- SYSTEM START: Recite Surah Al-Ikhlas now ---")

# Open the "Input Stream"
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("--- RECORDING COMPLETE: Saving to Disk ---")

stream.stop_stream()
stream.close()
audio.terminate()

# Write the data to a WAV file
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
