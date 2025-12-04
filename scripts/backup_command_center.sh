#!/usr/bin/env bash
# Simple backup helper for Command Center.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
BACKUP_NAME="command_center_$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Не найден файл окружения: $ENV_FILE" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
TMP_ROOT="$(mktemp -d)"
PAYLOAD_DIR="$TMP_ROOT/$BACKUP_NAME"
FILES_DIR="$PAYLOAD_DIR/files"
mkdir -p "$FILES_DIR"

# Загружаем переменные из .env (POSTGRES_*).
set -a
source "$ENV_FILE"
set +a

REQUIRED_VARS=(POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD)
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Переменная $var не задана в $ENV_FILE, бэкап невозможен" >&2
    rm -rf "$TMP_ROOT"
    exit 1
  fi
done

echo "→ Дамп PostgreSQL..."
DOCKER_COMPOSE="docker compose -f $PROJECT_ROOT/docker-compose.yml"
$DOCKER_COMPOSE exec -T postgres bash -c \
  "PGPASSWORD='${POSTGRES_PASSWORD}' pg_dump -U '${POSTGRES_USER}' '${POSTGRES_DB}'" \
  > "$PAYLOAD_DIR/postgres.sql"

copy_path() {
  local rel=$1
  local src="$PROJECT_ROOT/$rel"
  local dest="$FILES_DIR/$rel"
  if [[ -d "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
    echo "  • каталог $rel"
  elif [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
    echo "  • файл $rel"
  fi
}

echo "→ Копирование конфигов и данных..."
CONFIG_ITEMS=(
  ".env"
  ".env.mcp"
  ".env.mcp.example"
  "docker-compose.yml"
  "docker-compose.mcp.local.yml"
  "docker-compose.mcp.example.yml"
  "requirements.txt"
)
CONFIG_DIRS=(
  "docs"
  "media"
  "command_center/config"
)

for item in "${CONFIG_ITEMS[@]}"; do
  copy_path "$item"
done
for dir in "${CONFIG_DIRS[@]}"; do
  copy_path "$dir"
done

cat > "$PAYLOAD_DIR/backup_info.txt" <<EOF
Command Center backup
Дата:        $(date -Iseconds)
Хост:        $(hostname || echo "unknown")
ENV файл:    $ENV_FILE
Postgres DB: $POSTGRES_DB
Скрипт:      scripts/backup_command_center.sh
EOF

ARCHIVE_PATH="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo "→ Архивация в $ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$TMP_ROOT" "$BACKUP_NAME"

rm -rf "$TMP_ROOT"
echo "Готово. Архив: $ARCHIVE_PATH"
