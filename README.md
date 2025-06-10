# radio-decoder-starter

# SDR Decoder Starter (FastAPI)

Минимальный FastAPI-сервер для загрузки IQ-файлов и запуска на порту 8000.

## 🚀 Установка

```bash
# Установка Docker и Docker Compose
sudo apt update
sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker $USER
newgrp docker

# Клонирование проекта
git clone https://github.com/<your-repo>/radio-decoder-starter.git
cd radio-decoder-starter

# Сборка и запуск
docker-compose up --build -d

# Проверка
curl http://localhost:8000/
