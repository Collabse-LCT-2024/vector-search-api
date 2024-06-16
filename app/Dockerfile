# Стадия сборки
FROM python:3.9 AS build

WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем остальные файлы приложения
COPY . .

# Команда для запуска приложения
CMD ["fastapi", "run", "main.py", "--port", "80"]

