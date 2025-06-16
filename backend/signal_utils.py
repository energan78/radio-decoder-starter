import os
import numpy as np
import soundfile as sf

def load_signal(filepath):
    """
    Универсальная функция загрузки сигнала из файла.
    Поддерживает форматы: .npy (NumPy), .wav (аудио), .bin/.raw (complex64/float32).
    Возвращает массив numpy.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        data = np.load(filepath)
    elif ext == ".wav":
        data, _ = sf.read(filepath)
        # Если стерео — берём только первый канал
        if data.ndim > 1:
            data = data[:, 0]
    else:
        # Пробуем complex64, если не получилось — float32
        try:
            data = np.fromfile(filepath, dtype=np.complex64)
        except Exception:
            data = np.fromfile(filepath, dtype=np.float32)
    return data