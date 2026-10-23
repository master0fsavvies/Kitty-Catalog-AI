import os
import re
import time
import subprocess
from pathlib import Path
from pystray import Icon, MenuItem, Menu
from PIL import Image
import threading


from telegram import Update
from telegram.error import TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from kitty_ai import find_media


TOKEN_FILE = Path("token.txt")

def load_token():
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"Token file not found: {TOKEN_FILE}")
    return TOKEN_FILE.read_text(encoding="utf-8").strip()


TELEGRAM_BOT_TOKEN = load_token()

STREAMLIT_APP = "app.py"

SEND_TIMEOUT = 180
CONNECT_TIMEOUT = 30

streamlit_process = None


def start_streamlit():
    global streamlit_process

    # Kill previous Streamlit instance
    if streamlit_process is not None:
        try:
            if streamlit_process.poll() is None:
                streamlit_process.terminate()
                try:
                    streamlit_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    streamlit_process.kill()
        except Exception:
            pass

    streamlit_process = subprocess.Popen(
        [
            "streamlit",
            "run",
            STREAMLIT_APP
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    local_url = None
    network_url = None

    start_time = time.time()

    while time.time() - start_time < 20:
        line = streamlit_process.stdout.readline()

        if not line: continue

        print("[STREAMLIT]", line.strip())

        local_match = re.search(r"Local URL:\s*(http://\S+)", line)
        network_match = re.search(r"Network URL:\s*(http://\S+)", line)

        if local_match: local_url = local_match.group(1)

        if network_match: network_url = network_match.group(1)

        if local_url and network_url: break

    return local_url, network_url


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a texting situation, and I'll find the best kitty reactions.\n\n"
        "Send 'manual' to launch the manual Streamlit search."
    )


async def send_media_result(update: Update, item, rank):
    file_path = Path(item["file_path"])

    caption = (
        f"#{rank} — ID {item['id']}\n"
        f"Score: {item.get('score', 0):.3f}\n"
        f"{item.get('reason', '')}"
    ).strip()

    if not file_path.exists():
        await update.message.reply_text(f"{caption}\n\nMissing file:\n{file_path}")
        return

    media_type = item.get("media_type", "").lower()

    try:
        if media_type == "image":
            with open(file_path, "rb") as f:
                await update.message.reply_photo(
                    photo=f,
                    caption=caption,
                    read_timeout=SEND_TIMEOUT,
                    write_timeout=SEND_TIMEOUT,
                    connect_timeout=CONNECT_TIMEOUT,
                    pool_timeout=SEND_TIMEOUT,
                )

        elif media_type == "video":
            with open(file_path, "rb") as f:
                await update.message.reply_video(
                    video=f,
                    caption=caption,
                    supports_streaming=True,
                    read_timeout=SEND_TIMEOUT,
                    write_timeout=SEND_TIMEOUT,
                    connect_timeout=CONNECT_TIMEOUT,
                    pool_timeout=SEND_TIMEOUT,
                )

        else:
            with open(file_path, "rb") as f:
                await update.message.reply_document(
                    document=f,
                    caption=caption,
                    read_timeout=SEND_TIMEOUT,
                    write_timeout=SEND_TIMEOUT,
                    connect_timeout=CONNECT_TIMEOUT,
                    pool_timeout=SEND_TIMEOUT,
                )

    except TimedOut:
        await update.message.reply_text(f"{caption}\n\nUpload is taking longer than expected. It may still arrive.")

    except Exception as e:
        await update.message.reply_text(f"{caption}\n\nCould not send file: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.strip()

    if not prompt:
        await update.message.reply_text("Send me a prompt to search.")
        return

    if prompt.lower() == "manual":

        await update.message.reply_text("Launching manual Kitty Search...")

        local_url, network_url = start_streamlit()

        if local_url or network_url:
            message = []

            if local_url: message.append(f"Local URL:\n{local_url}")
            if network_url: message.append(f"Network URL:\n{network_url}")

            await update.message.reply_text("\n\n".join(message))

        else:
            await update.message.reply_text("Could not detect Streamlit URLs.")

        return

    await update.message.reply_text(f"Searching kitty library for:\n{prompt}")

    try: results = find_media(prompt, limit=10)

    except Exception as e:
        await update.message.reply_text(f"Search failed: {e}")
        return

    if not results:
        await update.message.reply_text("No matches found.")
        return

    for rank, item in enumerate(results[:3], start=1):
        await send_media_result(update, item, rank)

def create_image():
    return Image.open("kitty.ico")


def quit_app(icon, item):
    icon.stop()
    os._exit(0)


def run_tray():
    icon = Icon(
        "Kitty AI",
        create_image(),
        menu=Menu(MenuItem("Exit", quit_app))
    )

    icon.run()


def main():
    threading.Thread(target=run_tray, daemon=True).start()

    
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Missing Telegram bot token. Please fill a token.txt file with your bot token.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Kitty Telegram bot is running.")

    app.run_polling()


if __name__ == "__main__":
    main()