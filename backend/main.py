import os
import json
import re
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
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
        return {}

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
async def add_signal(request: AddSignalRequest, file: UploadFile = File(...)):
    try:
        class_dir = os.path.join(SIGNAL_LIBRARY_PATH, request.signal_type)
        os.makedirs(class_dir, exist_ok=True)
        file_path = os.path.join(class_dir, file.filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await file.read())
        logger.info(f"Файл {file.filename} добавлен в класс {request.signal_type}")
        return {"message": f"Файл добавлен в класс {request.signal_type}"}
    except Exception as e:
        logger.error(f"Ошибка добавления сигнала: {e}")
        raise HTTPException(status_code=500, detail="Ошибка добавления сигнала")

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
