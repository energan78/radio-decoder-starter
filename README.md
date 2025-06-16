# Radio Decoder Starter

## Возможности

- Классификация радиосигналов (FM, AM, GSM, WiFi и др.) с помощью нейросети (CNN)
- Самообучение на новых данных и интеграция с датасетами RadioML (2016, 2018)
- Загрузка и хранение пользовательских примеров сигналов
- REST API для загрузки, классификации и пополнения библиотеки сигналов
- Онлайн и офлайн распознавание речи из аудиофайлов (Google Speech и Vosk)
- Загрузка сигналов разных форматов
- Визуализация временного ряда, спектра, вероятностей, активаций слоёв, матрицы неточностей
- Сохранение сигнала с комментарием
---

## Установка и запуск (Ubuntu)

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
   [Kaggle RadioML 2018.01A](https://www.kaggle.com/datasets/pinxau1000/radioml2018)  
   Файл: `RML2018.01A.h5`

2. **Создайте папку и поместите файл**  
   ```bash
   mkdir -p backend/radioml2018
   mv /path/to/RML2018.01A.h5 backend/radioml2018/
   ```

3. **Путь к датасету можно изменить через веб-интерфейс или в `backend/config.json`.**

---

## Интеграция офлайн-распознавания речи (Vosk)

1. Скачайте русскую модель Vosk:  
   [vosk-model-small-ru-0.22.zip](https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip)
2. Распакуйте в папку `backend/vosk-model-ru/`
3. Путь к модели можно изменить через веб-интерфейс или в `backend/config.json`.

---

## Запуск сервера

**Внимание:** Всегда активируйте виртуальное окружение перед запуском!

```bash
uvicorn backend.main:app --reload
```

- Сервер будет доступен по адресу: http://<ваш_IP>:8000
- Документация API: http://<ваш_IP>:8000/docs
- Веб-интерфейс управления: http://<ваш_IP>:8000/settings_ui

---

## Управление настройками через веб-интерфейс

Откройте [http://<ваш_IP>:8000/settings_ui](http://<ваш_IP>:8000/settings_ui)  
Вы сможете включать/отключать функции, менять пути к моделям и датасетам, настраивать параметры AI-модулей.

---

## Дополнительные веб-страницы

- **Главная страница:** http://<ваш_IP>:8000/
- **Загрузка сигналов:** http://<ваш_IP>:8000/upload_ui
- **Просмотр логов:** http://<ваш_IP>:8000/logs_ui
- **Состояние сервера:** http://<ваш_IP>:8000/status_ui

---

## Примеры запросов к API

- **Загрузка файла:**  
  `POST /upload`
- **Классификация сигнала:**  
  `POST /classify_signal`
- **Добавление сигнала в библиотеку:**  
  `POST /add_signal`
- **Онлайн-распознавание речи (Google):**  
  `POST /recognize_speech`
- **Офлайн-распознавание речи (Vosk):**  
  `POST /recognize_speech_offline`
- **Детектор аномалий:**  
  `POST /detect_anomaly`
- **Геоклассификация:**  
  `POST /classify_by_geo`
- **Обучение модели:**  
  `POST /train_model`

---

## Пример запроса для офлайн-распознавания речи (curl)

```bash
curl -X POST "http://<ваш_IP>:8000/recognize_speech_offline" \
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
- Все основные параметры и функции можно настраивать через веб-интерфейс!
- **Перед каждым запуском сервера не забывайте активировать виртуальное окружение:**
  ```bash
  source venv/bin/activate
  ```

---

## Обновление проекта

Если вы уже устанавливали проект ранее и хотите получить последние изменения из репозитория, выполните следующие шаги:

### 1. Перейдите в папку проекта

```bash
cd /путь/к/вашему/radio-decoder-starter
```

### 2. Сохраните свои локальные изменения (если есть)

Если вы что-то меняли локально и не хотите потерять изменения:

```bash
git add .
git commit -m "Мои локальные изменения"
```

### 3. Получите последние изменения с GitHub

```bash
git pull origin main
```

Если появятся сообщения о конфликтах или неотслеживаемых файлах:
- Добавьте нужные файлы в git:  
  `git add <имя_файла>`
- Удалите ненужные файлы:  
  `rm <имя_файла>`
- После этого снова выполните `git pull origin main`.

### 4. Обновите зависимости

Активируйте виртуальное окружение:

```bash
source venv/bin/activate
```

Установите новые зависимости:

```bash
pip install -r backend/requirements.txt
```

### 5. Перезапустите сервер

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

**Теперь ваше приложение обновлено до последней версии из репозитория!**

---

## Быстрая команда для обновления (если нет локальных изменений)

```bash
cd /путь/к/вашему/radio-decoder-starter
git pull origin main
source venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

Если появятся ошибки — смотрите сообщения git или pip, либо обратитесь за помощью!
