import asyncio
import signal
import warnings
import logging
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

# Remove Terminal User warning
warnings.filterwarnings('ignore', category=UserWarning)
#logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
# Vars
sample_rate = 44100  
chunk_duration = 5

audio_queue = []
async def record_audio():
    print("Recording audio...")
    recording = await asyncio.to_thread(sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16"))
    sd.wait()
    audio_queue.append(recording)

async def write_audio():
    if audio_queue: 
        print("Saving audio to file...")
        recording = audio_queue.pop(0)
        wav.write("output_audio.wav", sample_rate, recording)
    else:
        return
    
async def transcribe():
    print("Transcribing audio...")
    model = whisper.load_model("tiny")
    result = model.transcribe("output_audio.wav")
    print(result["text"])

async def shutdown(signal, loop):
    logging.info(f"Recieved exit signal {signal.name} ... ")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]
    logging.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Flushing metrics")
    loop.stop()

async def main():
    loop = asyncio.get_event_loop()

    for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    while True:
        try: 
            await record_audio()
            await write_audio()
            await transcribe()
        except asyncio.CancelledError:
            break

        finally:
            logging.info("Successfully shutdown Hermes service")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Manual interrupt recieved. Exiting...")

