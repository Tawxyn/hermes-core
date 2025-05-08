from ring_buffer import RingBuffer
import asyncio
import os
import signal
import time
import glob
import warnings
import logging
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

# Current directory 
current_directory = os.getcwd()
# Remove Terminal User warning
warnings.filterwarnings('ignore', category=UserWarning)
#logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
# Vars
sample_rate = 44100  
chunk_duration = 2
# Queues
audio_queue = asyncio.Queue(maxsize=10)
transcribe_queue = asyncio.Queue(maxsize=10)

# Model
model = whisper.load_model("tiny")

buffer = RingBuffer(max_chunks=5)

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
        try:
            await record_audio()
        except:
            asyncio.CancelledError
            logging.info("✅ record_loop successfully interupted")
            break

# ---------- WRITE ----------
async def write_audio():
    recording = await audio_queue.get()
    filename = f"output_audio{int(time.time())}.wav"
    wav.write(filename, sample_rate, recording)
    await transcribe_queue.put(filename)

async def write_loop():
    while True:
        try:
            await write_audio()
        except:
            asyncio.CancelledError
            logging.info("✅ write_loop successfully interupted")
            break

# ---------- TRANSCRIBE ----------
async def transcribe():
    filename = await transcribe_queue.get()
    result = model.transcribe(filename)
    print(result["text"])

async def transcribe_loop():
    while True:
        try:
            await transcribe()
        except:
            asyncio.CancelledError
            logging.info("✅ transcribe_loop successfully interupted")
            break

#  ---------- POSTRUN ----------

def delete_wav_files(current_directory):
    wav_files = glob.glob(os.path.join(current_directory, "*.wav"))
    i = 0
    for file_path in wav_files:
        try:
            os.remove(file_path)
            i += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    logging.info(f"Deleted: {i} .wav files")

async def shutdown(signal, loop):
    logging.info(f"Recieved exit signal {signal.name} ... ")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    # Cancel all tasks
    [task.cancel() for task in tasks]
    logging.info(f"Cancelling {len(tasks)} outstanding tasks")

    # Wait for tasks to be all cancled
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Flushing metrics")
    loop.stop()

async def main():
    loop = asyncio.get_event_loop()

    for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))
    
    try:
        record_task = asyncio.create_task(record_loop())
        write_task = asyncio.create_task(write_loop())
        transcribe_task = asyncio.create_task(transcribe_loop())
        await asyncio.gather(record_task, write_task, transcribe_task)
    except:
        asyncio.CancelledError
        logging.info("✅ asyncio tasks successfully interupted")

    delete_wav_files(current_directory)
    logging.info("Successfully shutdown hermes services")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Manual interrupt recieved. Exiting...")

