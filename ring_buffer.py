from collections import deque 
import numpy as np

class RingBuffer:
    def __init__(self, max_chunks: int):
        self.buffer = deque(maxlen=max_chunks)

    def add_chunks(self, chunk: np.ndarray):
        self.buffer.append(chunk)

    def get_concatenated(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=0)

    def clear(self):
        self.buffer.clear()
