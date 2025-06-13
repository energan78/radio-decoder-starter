# Radio Decoder Starter

## Возможности

- Классификация радиосигналов (FM, AM, GSM, WiFi и др.) с помощью нейросети (CNN)
- Самообучение на новых данных и интеграция с датасетами RadioML (2016, 2018)
- Загрузка и хранение пользовательских примеров сигналов
- REST API для загрузки, классификации и пополнения библиотеки сигналов
- Онлайн и офлайн распознавание речи из аудиофайлов (Google Speech и Vosk)
- Детектор аномалий, геоклассификация, анализ эмоций в речи
- Веб-интерфейс для управления всеми функциями и настройками

---

## Быстрый старт: Установка на Ubuntu

### 1. Установите системные зависимости

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip ffmpeg git
```

### 2. Клонируйте репозиторий

```bash
git clone https://github.com/energan78/radio-decoder-starter.git
cd radio-decoder-starter
```

### 3. Создайте и активируйте виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Установите Python-зависимости

```bash
pip install -r backend/requirements.txt
```

---

## Интеграция RadioML 2018.01A

1. **Скачайте датасет RadioML 2018.01A**  
   Перейдите на [Kaggle RadioML 2018.01A](https://www.kaggle.com/datasets/pinxau1000/radioml2018) и скачайте файл `RML2018.01A.h5`.

2. **Создайте папку для датасета и поместите файл**  
   ```bash
   mkdir -p backend/radioml2018
   mv /path/to/RML2018.01A.h5 backend/radioml2018/
   ```

3. **Путь к датасету можно изменить через веб-интерфейс или в `backend/config.json`.**

---

## Обучение модели

```bash
python3 backend/train_signal_model.py
```

---

## Использование офлайн-распознавания речи

1. Скачайте русскую модель Vosk с https://alphacephei.com/vosk/models  
   Например, [vosk-model-small-ru-0.22.zip](https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip)
2. Распакуйте её в папку `backend/vosk-model-ru/`
3. Путь к модели можно изменить через веб-интерфейс или в `backend/config.json`.

---

## Запуск сервера

```bash
uvicorn backend.main:app --reload
```

- Сервер будет доступен по адресу: http://localhost:8000
- Документация API: http://localhost:8000/docs
- **Веб-интерфейс управления:** http://localhost:8000/settings_ui

---

## Управление настройками через веб-интерфейс

Откройте [http://localhost:8000/settings_ui](http://localhost:8000/settings_ui)  
Вы сможете включать/отключать функции, менять пути к моделям и датасетам, настраивать параметры AI-модулей.

---

## Примеры запросов к API

- **Загрузка файла:**
  ```
  POST /upload
  ```
- **Классификация сигнала:**
  ```
  POST /classify_signal
  ```
- **Добавление сигнала в библиотеку:**
  ```
  POST /add_signal
  ```
- **Онлайн-распознавание речи (Google):**
  ```
  POST /recognize_speech
  ```
- **Офлайн-распознавание речи (Vosk):**
  ```
  POST /recognize_speech_offline
  ```
- **Детектор аномалий:**
  ```
  POST /detect_anomaly
  ```
- **Геоклассификация:**
  ```
  POST /classify_by_geo
  ```

---

## Пример запроса для офлайн-распознавания речи (curl)

```bash
curl -X POST "http://localhost:8000/recognize_speech_offline" \
  -H  "accept: application/json" \
  -H  "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.wav;type=audio/wav"
```

---

## Примечания

- Для работы с аудиофайлами требуется установленный ffmpeg:  
  `sudo apt-get install ffmpeg`
- Для офлайн-распознавания речи требуется модель Vosk (см. выше).
- Для онлайн-распознавания речи требуется интернет-соединение.
- Для обучения на RadioML 2018.01A требуется файл `RML2018.01A.h5` в папке `backend/radioml2018/`.
- Все основные параметры и функции можно настраивать через веб-интерфейс по адресу [http://localhost:8000/settings_ui](http://localhost:8000/settings_ui).
