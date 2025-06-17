import os
import json
import re
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, HttpUrl
import httpx
import aiofiles
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from backend.train_signal_model import ConvSignalNet, SAMPLE_LEN
import speech_recognition as sr
from fastapi import UploadFile, File
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import wave
import json
import subprocess
from backend.signal_utils import (
    load_signal, extract_freq_from_filename, get_class_stats, create_class_folder
)
from backend.match_freq_band import match_frequency
from fastapi import Body
from backend.analyzer import analyze_signal_file
import joblib
from backend.freq_db import FREQ_DB

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Radio Decoder API",
    description="API для загрузки и обработки радиофайлов",
    version="1.0.0"
)

class FetchRequest(BaseModel):
    url: HttpUrl

def load_config():
    try:
        with open("backend/config.json") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка при чтении config.json: {e}")
        # Значения по умолчанию
        return {
            "use_anomaly_detector": False,
            "use_geo_classifier": False
        }

config = load_config()

UPLOAD_FOLDER = "backend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SIGNAL_LIBRARY_PATH = "backend/signal_library"
os.makedirs(SIGNAL_LIBRARY_PATH, exist_ok=True)

def extract_frequency(filename):
    match = re.search(r"(\d{3,4})MHz", filename)
    if match:
        return match.group(1)
    return "unknown"

# Глобальный обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Ошибка: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"}
    )

@app.post("/upload", summary="Загрузить файл", description="Загрузка радиофайла и извлечение частоты из имени файла.")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        async with aiofiles.open(file_location, "wb") as f:
            content = await file.read()
            await f.write(content)
        freq = extract_frequency(file.filename)
        logger.info(f"Файл {file.filename} загружен. Частота: {freq}")
        return {"filename": file.filename, "frequency": freq}
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при сохранении файла")

@app.post("/fetch", summary="Получить содержимое по URL", description="Асинхронно получить содержимое по переданному URL.")
async def fetch_url(request: FetchRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(request.url)
        response.raise_for_status()
        logger.info(f"Успешно получено содержимое по URL: {request.url}")
        return {"content": response.text}
    except Exception as e:
        logger.error(f"Ошибка при получении URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Пример простой нейросети для классификации сигналов
class SimpleSignalNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSignalNet, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Заглушка: список классов (можно расширять)
def get_signal_classes():
    classes = []
    for d in os.listdir(SIGNAL_LIBRARY_PATH):
        if os.path.isdir(os.path.join(SIGNAL_LIBRARY_PATH, d)):
            classes.append(d)
    return classes if classes else ["FM", "AM", "GSM", "WiFi"]

MODEL_PATH = "backend/signal_model.pth"

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    model = ConvSignalNet(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes

# Пример функции для классификации сигнала
def classify_signal_with_model(iq_data: np.ndarray):
    model, classes = load_model()
    # Преобразуем IQ-данные к фиксированной длине (например, 1024)
    if len(iq_data) < 1024:
        iq_data = np.pad(iq_data, (0, 1024 - len(iq_data)), 'constant')
    else:
        iq_data = iq_data[:1024]
    x = np.abs(iq_data).astype(np.float32)
    x = torch.tensor(x).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy()[0]
        idx = np.argmax(probs)
        return classes[idx], float(probs[idx])

class SignalTypeResponse(BaseModel):
    signal_type: str
    confidence: float

@app.post("/classify_signal", response_model=SignalTypeResponse, summary="Классификация типа радиосигнала")
async def classify_signal(file: UploadFile = File(...)):
    try:
        content = await file.read()
        iq_data = np.frombuffer(content, dtype=np.complex64)
        signal_type, confidence = classify_signal_with_model(iq_data)
        logger.info(f"Файл {file.filename} классифицирован как {signal_type} ({confidence:.2f})")
        return SignalTypeResponse(signal_type=signal_type, confidence=confidence)
    except Exception as e:
        logger.error(f"Ошибка классификации сигнала: {e}")
        raise HTTPException(status_code=500, detail="Ошибка классификации сигнала")

# Эндпоинт для добавления нового сигнала в библиотеку
class AddSignalRequest(BaseModel):
    signal_type: str

@app.post("/add_signal", summary="Добавить пример сигнала в библиотеку")
async def add_signal(
    file: UploadFile = File(...),
    signal_type: str = Form(...),
    comment: str = Form(None)
):
    save_dir = f"backend/signal_library/{signal_type}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)
    with open(save_path, "wb") as f_out:
        f_out.write(await file.read())
    if comment:
        with open(save_path + ".txt", "w", encoding="utf-8") as f_comment:
            f_comment.write(comment)
    from backend.signal_utils import load_signal
    data = load_signal(save_path)
    info = {
        "length": len(data),
        "dtype": str(data.dtype)
    }
    # Определяем частоту и генерируем комментарий
    freq_mhz = extract_freq_from_filename(file.filename)
    if freq_mhz is not None:
        # далее используйте freq_mhz для match_frequency и генерации комментария
        comment_data = match_frequency(freq_mhz)
        comment = f"{comment_data['label']} — {comment_data['usage']} (модуляция: {comment_data['modulation']})"

    return {"status": "ok", "info": info}

VOSK_MODEL_PATH = "backend/vosk-model-ru"

class SpeechRecognitionResponse(BaseModel):
    text: str
    comment: str

@app.post("/recognize_speech", response_model=SpeechRecognitionResponse, summary="Распознавание речи из аудиофайла")
async def recognize_speech(file: UploadFile = File(...)):
    try:
        # Сохраняем файл во временный WAV
        temp_path = f"/tmp/{file.filename}"
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(await file.read())

        # Если файл не WAV, конвертируем в WAV
        if not temp_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(temp_path)
            temp_wav = temp_path + ".wav"
            audio.export(temp_wav, format="wav")
            temp_path = temp_wav

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")
            comment = "Обнаружена речь" if text.strip() else "Речь не обнаружена"
        except sr.UnknownValueError:
            text = ""
            comment = "Речь не обнаружена"
        except Exception as e:
            text = ""
            comment = f"Ошибка распознавания: {e}"

        logger.info(f"Распознанный текст: {text}")
        return SpeechRecognitionResponse(text=text, comment=comment)
    except Exception as e:
        logger.error(f"Ошибка распознавания речи: {e}")
        raise HTTPException(status_code=500, detail="Ошибка распознавания речи")

@app.post("/recognize_speech_offline", response_model=SpeechRecognitionResponse, summary="Офлайн-распознавание речи из аудиофайла")
async def recognize_speech_offline(file: UploadFile = File(...)):
    try:
        # Сохраняем файл во времний WAV
        temp_path = f"/tmp/{file.filename}"
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(await file.read())

        # Если файл не WAV, конвертируем в WAV
        if not temp_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(temp_path)
            temp_wav = temp_path + ".wav"
            audio.export(temp_wav, format="wav")
            temp_path = temp_wav

        # Открываем WAV-файл
        wf = wave.open(temp_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 44100]:
            # Преобразуем в нужный формат
            audio = AudioSegment.from_file(temp_path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            temp_fixed = temp_path + ".fixed.wav"
            audio.export(temp_fixed, format="wav")
            wf = wave.open(temp_fixed, "rb")

        # Загружаем модель Vosk
        if not os.path.exists(VOSK_MODEL_PATH):
            raise Exception("Vosk model not found!")
        model = Model(VOSK_MODEL_PATH)
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        # Распознаём аудио
        result_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                part = json.loads(rec.Result())
                result_text += part.get("text", "") + " "
        # Финальный результат
        part = json.loads(rec.FinalResult())
        result_text += part.get("text", "")

        result_text = result_text.strip()
        comment = "Обнаружена речь" if result_text else "Речь не обнаружена"

        logger.info(f"Vosk офлайн-распознавание: {result_text}")
        return SpeechRecognitionResponse(text=result_text, comment=comment)
    except Exception as e:
        logger.error(f"Ошибка офлайн-распознавания речи: {e}")
        raise HTTPException(status_code=500, detail="Ошибка офлайн-распознавания речи")

@app.get("/signal_classes")
def signal_classes():
    stats = get_class_stats()
    return {"classes": list(stats.keys())}

@app.post("/create_class_folder")
def create_class_folder_api(class_name: str = Body(..., embed=True)):
    create_class_folder(class_name)
    return {"status": "ok"}

@app.post("/analyze_signal")
async def analyze_signal(
    file: UploadFile = File(...),
    signal_type: str = Form(...),
    comment: str = Form(None)
):
    temp_path = f"backend/temp/{file.filename}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f_out:
        f_out.write(await file.read())

    freq_mhz = extract_freq_from_filename(file.filename)
    band_info = match_frequency(freq_mhz) if freq_mhz else None
    comment_auto = None
    if band_info:
        comment_auto = f"{band_info['label']} — {band_info['usage']} (модуляция: {band_info['modulation']})"

    result = analyze_signal_file(
        temp_path,
        model_pytorch=model_pytorch,
        model_rf=model_rf,
        model_svm=model_svm,
        sample_len=sample_len
    )

    return {
        **result,
        "freq_mhz": freq_mhz,
        "label": band_info["label"] if band_info else None,
        "modulation": band_info["modulation"] if band_info else None,
        "usage": band_info["usage"] if band_info else None,
        "comment_auto": comment_auto,
    }

@app.get("/settings")
def get_settings():
    return config

@app.post("/settings")
def update_settings(new_settings: dict):
    config.update(new_settings)
    # Сохраняем изменения в файл
    with open("backend/config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config

@app.get("/settings_ui", response_class=HTMLResponse)
def settings_ui():
    return FileResponse("backend/settings.html")

@app.get("/logs_ui", response_class=HTMLResponse)
def logs_ui():
    return FileResponse("backend/logs_ui.html")

@app.get("/upload_ui", response_class=HTMLResponse)
def upload_ui():
    return FileResponse("backend/upload_ui.html")

@app.get("/logs")
def get_logs():
    log_path = "backend/app.log"
    if not os.path.exists(log_path):
        return "Лог-файл не найден."
    with open(log_path, "r", encoding="utf-8") as f:
        return f.read()[-10000:]  # последние 10к символов

@app.get("/status_ui", response_class=HTMLResponse)
def status_ui():
    return FileResponse("backend/status_ui.html")

@app.get("/signal_ui", response_class=HTMLResponse)
def signal_ui():
    """
    Возвращает HTML-страницу для работы с сигналом.
    """
    return FileResponse("backend/signal_ui.html")

@app.get("/status")
def status():
    # Пример проверки состояния
    model_loaded = os.path.exists("backend/signal_model.pth")
    vosk_model = os.path.exists(config.get("vosk_model_path", "backend/vosk-model-ru"))
    radioml_dataset = os.path.exists(config.get("radioml_dataset_path", "backend/radioml2018/RML2018.01A.h5"))
    errors = ""
    return {
        "model_loaded": model_loaded,
        "vosk_model": vosk_model,
        "radioml_dataset": radioml_dataset,
        "errors": errors
    }

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Radio Decoder Starter</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f4f8; }
            .container { background: #fff; max-width: 520px; margin: 40px auto; padding: 36px 28px 28px 28px; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.10);}
            h1 { text-align: center; color: #1976d2; margin-bottom: 24px; }
            ul { list-style: none; padding: 0; }
            li { margin: 18px 0; }
            a, button { display: block; width: 100%; text-align: center; background: #1976d2; color: #fff; text-decoration: none; padding: 14px 0; border-radius: 8px; font-size: 1.1em; margin-bottom: 10px; border: none; cursor: pointer; transition: background 0.2s;}
            a:hover, button:hover { background: #125ea8; }
            .desc { text-align: center; color: #555; margin-bottom: 24px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Radio Decoder Starter</h1>
            <div class="desc">Многофункциональный сервис для анализа радиосигналов и работы с ИИ</div>
            <ul>
                <li><a href="/settings_ui">⚙️ Управление настройками</a></li>
                <li><a href="/docs">📚 Документация API (Swagger)</a></li>
                <li><a href="/signal_ui">🎛 Работа с сигналом</a></li>
                <li><a href="/upload_ui">⬆️ Загрузка новых сигналов</a></li>
                <li><a href="/logs_ui">📝 Просмотр логов</a></li>
                <li><a href="/status_ui">📊 Состояние сервера</a></li>
            </ul>
        </div>
    </body>
    </html>
    """

if config.get("use_anomaly_detector", False):
    # использовать config["anomaly_threshold"] при обработке
    pass

if config.get("use_speech_recognition", False):
    # включить/выключить распознавание речи
    pass

@app.post("/train_model")
def train_model():
    # Запуск обучения в фоне (можно доработать под вашу ОС)
    subprocess.Popen(["python3", "backend/train_signal_model.py"])
    return {"status": "training started"}

# Загрузка моделей (один раз при старте)
import torch
from backend.train_signal_model import ConvSignalNet, SAMPLE_LEN

@app.get("/signal_stats")
def signal_stats():
    """
    Возвращает статистику по классам сигналов для интерфейса.
    """
    return get_class_stats()

# Загрузка PyTorch-модели
try:
    model_data = torch.load("backend/signal_model.pth", map_location="cpu")
    model_pytorch = ConvSignalNet(num_classes=len(model_data["classes"]))
    model_pytorch.load_state_dict(model_data["model_state"])
    model_pytorch.eval()
    model_pytorch.classes = model_data["classes"]
except Exception:
    model_pytorch = None

# Загрузка RF и SVM моделей
try:
    model_rf = joblib.load("backend/rf_model.pkl")
except Exception:
    model_rf = None

try:
    model_svm = joblib.load("backend/svm_model.pkl")
except Exception:
    model_svm = None

sample_len = 1024  # или SAMPLE_LEN, если он определён в train_signal_model.py

# TODO: Определить частоту файла, сопоставить с базой диапазонов
# и автоматически сгенерировать человекочитаемый комментарий о типе сигнала.

def find_band_by_freq(freq):
    for band in FREQ_DB:
        if band["start"] <= freq <= band["end"]:
            return band
    return None
