import numpy as np
from backend.signal_utils import load_signal

def analyze_signal_file(filepath, model_pytorch=None, model_rf=None, model_svm=None, sample_len=1024):
    """
    Анализирует сигнал из файла:
    - вычисляет частоту дискретизации (если возможно)
    - строит спектр
    - классифицирует с помощью моделей (PyTorch, RF, SVM)
    Возвращает словарь с результатами.
    """
    data = load_signal(filepath)
    result = {}

    # Временной ряд (нормализованный)
    result["waveform"] = (data[:sample_len] / np.max(np.abs(data[:sample_len]))).tolist()

    # Спектр
    spectrum = np.abs(np.fft.fft(data[:sample_len]))
    result["spectrum"] = (spectrum / np.max(spectrum)).tolist()

    # Частота дискретизации (если WAV)
    try:
        import soundfile as sf
        info = sf.info(filepath)
        result["samplerate"] = info.samplerate
    except Exception:
        result["samplerate"] = None

    # Классификация PyTorch
    if model_pytorch is not None:
        import torch
        x = np.abs(data[:sample_len]).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)
        out = model_pytorch(x)
        probs = torch.softmax(out, dim=1).detach().cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))
        result["pytorch_probs"] = probs
        result["pytorch_pred"] = model_pytorch.classes[pred_idx] if hasattr(model_pytorch, "classes") else str(pred_idx)

    # Классификация Random Forest
    if model_rf is not None:
        X_rf = np.abs(data[:sample_len]).reshape(1, -1)
        pred_rf = model_rf.predict(X_rf)[0]
        result["rf_pred"] = pred_rf

    # Классификация SVM
    if model_svm is not None:
        X_svm = np.abs(data[:sample_len]).reshape(1, -1)
        pred_svm = model_svm.predict(X_svm)[0]
        result["svm_pred"] = pred_svm

    return result