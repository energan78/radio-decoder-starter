import os
import numpy as np
import soundfile as sf

def load_signal(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        data = np.load(filepath)
    elif ext == ".wav":
        data, _ = sf.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]
    else:
        try:
            data = np.fromfile(filepath, dtype=np.complex64)
        except Exception:
            data = np.fromfile(filepath, dtype=np.float32)
    return data