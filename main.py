import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
# Set the sample rate (e.g., 44100 Hz)
fs = 44100  

# Set the duration of the recording (5.5 seconds)
duration = 10  

# Start recording
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

wav.write("output_audio.wav", fs, myrecording)
#model = whisper.load_model("tiny")
#result = model.transcribe("audio.mp3")
#print(result["text"])
