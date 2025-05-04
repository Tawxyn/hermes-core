import asyncio
import signal
import time
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
# Queues
audio_queue = asyncio.Queue(maxsize=10)
transcribe_queue = asyncio.Queue(maxsize=10)

# Model
model = whisper.load_model("tiny")

# ---------- RECORD ----------
async def record_audio():
    print("Recording audio...")
    recording = await asyncio.to_thread(record_block)
    await audio_queue.put(recording)

def record_block():
    recording = sd.rec(int(chunk_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    return recording

async def record_loop():
    while True:
        await record_audio()

# ---------- WRITE ----------
async def write_audio():
    recording = await audio_queue.get()
    filename = f"output_audio{int(time.time())}.wav"
    wav.write(filename, sample_rate, recording)
    await transcribe_queue.put(filename)
    logging.info("Saved and queued {filename} ...")

async def write_loop():
    while True:
        await write_audio()

# ---------- TRANSCRIBE ----------
async def transcribe():
    filename = await transcribe_queue.get()
    logging.info("Transcribing {filename} ... ")
    result = model.transcribe(filename)
    print(result["text"])

async def transcribe_loop():
    while True:
        await transcribe()


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
   
    record_task = asyncio.create_task(record_loop())
    write_task = asyncio.create_task(write_loop())
    transcribe_task = asyncio.create_task(transcribe_loop())

    await asyncio.gather(record_task, write_task, transcribe_task)
    
    logging.info("Successfully shutdown hermes services")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Manual interrupt recieved. Exiting...")

