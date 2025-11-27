FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Общие системные зависимости, нужные для MCP-серверов и самого проекта:
# - build-essential: для сборки Python-зависимостей
# - git: для pip install git+https://...
# - curl: для тестов/healthcheck'ов и потенциальных MCP
# - nodejs, npm: для MCP-серверов на Node (официальные mcpservers.org)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-зависимости проекта
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY . /app/

ENV DJANGO_SETTINGS_MODULE=command_center.settings

# Запуск Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
