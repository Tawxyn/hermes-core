import asyncio
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

# Vars
sample_rate = 44100  
chunk_duration = 5
device_index = 2

def record_audio():
    print("Recording audio...")
    return sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, device=device_index)

async def write_audio(myrecording):
    print("Saving audio to file...")
    wav.write("output_audio.wav", sample_rate, myrecording)
    
async def transcribe():
    print("Transcribing audio...")
    model = whisper.load_model("tiny")
    result = model.transcribe("output_audio.wav")
    print(result["text"])

async def main():
    while True:
        
        myrecording = await asyncio.to_thread(record_audio)
        
        await write_audio(myrecording)
        
        await transcribe()
        
        await asyncio.sleep(1)
        
asyncio.run(main())