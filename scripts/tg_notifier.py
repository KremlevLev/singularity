import json
import urllib.request
import os

# Пытаемся прочитать переменные из окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

# Локально в VS Code подгружаем .env, если библиотека установлена
if not TELEGRAM_BOT_TOKEN:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")
    except ImportError:
        pass

def send_telegram_notification(text: str):
    """Отправляет текстовое уведомление в Telegram."""
    url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_USER_ID, "text": text, "parse_mode": "Markdown"}

    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers
    )

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print("Уведомление в Telegram успешно отправлено!")
            else:
                print(f"Ошибка отправки Telegram: статус {response.status}")
    except Exception as e:
        print(f"Не удалось отправить уведомление в Telegram: {e}")


# Отправляем сигнал о старте скрипта
send_telegram_notification(
    "🚀 *Скрипт на Kaggle успешно запущен!*\n"
    "Очередь пройдена, TPU инициализирован. Начинаю подготовку к обучению модели Qwen."
)
