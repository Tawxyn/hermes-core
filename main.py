import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

print(sd.query_devices())
# Vars
sample_rate = 44100  
chunk_duration = 5  

while (True):
    myrecording = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    wav.write("output_audio.wav", sample_rate, myrecording)
    model = whisper.load_model("tiny")
    result = model.transcribe("audio.mp3")
    print(result["text"])
