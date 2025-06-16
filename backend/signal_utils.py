import os
import re
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

def extract_freq_from_filename(filename):
    """
    Извлекает частоту в МГц из имени файла, например: gsm_936.wav -> 936.0
    """
    match = re.search(r'(\d+(?:\.\d+)?)', filename)
    if match:
        return float(match.group(1))
    return None

def get_class_stats(base_dir="backend/signal_library"):
    """
    Возвращает словарь: {класс: количество файлов}
    """
    import os
    stats = {}
    for class_name in sorted(os.listdir(base_dir)):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len([f for f in os.listdir(class_dir) if f.endswith((".npy", ".wav", ".bin", ".raw"))])
        stats[class_name] = count
    return stats