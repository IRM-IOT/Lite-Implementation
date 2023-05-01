import pyaudio
import wave
import threading

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OVERLAP_RATIO = 0.5

frames = []

def record_audio(audioIndex):
    i = 1
    global frames
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # Check if we have enough frames for a clip
        if len(frames) == int(RATE * RECORD_SECONDS / CHUNK):
            name = 'audios/audio_' + str(i) + '.wav'
            # Create a new thread to handle writing the clip to disk
            clip_thread = threading.Thread(target=write_clip, args=(frames,name))
            clip_thread.start()

            # Remove the frames that have been used in the clip
            frames = frames[int(RATE * RECORD_SECONDS * (1 - OVERLAP_RATIO) / CHUNK):]
            
            audioIndex.append(name)
            i = i+1

def write_clip(frames,name):
    audio = pyaudio.PyAudio()
    wave_file = wave.open(name, "wb")
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b"".join(frames))
    wave_file.close()

# # Start recording in a new thread
# record_thread = threading.Thread(target=record_audio)
# record_thread.start()

record_audio([])
