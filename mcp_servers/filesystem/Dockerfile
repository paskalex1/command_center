FROM node:22-alpine

WORKDIR /app

# Нужен git, чтобы клонировать репозиторий
RUN apk add --no-cache git

# Клонируем репозиторий и ставим зависимости
RUN git clone https://github.com/cyanheads/filesystem-mcp-server.git . \
  && npm install \
  && npm run build

# По умолчанию настройки транспорта задаём через env (в docker-compose.mcp.local.yml)
# Здесь только дефолты на всякий случай
ENV MCP_TRANSPORT_TYPE=http \
    MCP_HTTP_PORT=8020 \
    MCP_HTTP_HOST=0.0.0.0 \
    MCP_LOG_LEVEL=info

# dist/index.js — как в официальной доке
CMD ["node", "dist/index.js"]
