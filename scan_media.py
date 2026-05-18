import os
import sqlite3
import hashlib

DB_PATH = "catlib.db"
MEDIA_FOLDER = "media"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}

def get_file_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

def get_media_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"

    return None

def scan_media_folder():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    added_count = 0
    skipped_count = 0

    for root, dirs, files in os.walk(MEDIA_FOLDER):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            media_type = get_media_type(file_path)

            if media_type is None:
                continue

            file_hash = get_file_hash(file_path)

            try:
                cursor.execute("""
                    INSERT INTO media_items (
                        file_path,
                        file_name,
                        media_type,
                        file_hash
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    file_path,
                    file_name,
                    media_type,
                    file_hash
                ))

                added_count += 1

            except sqlite3.IntegrityError:
                skipped_count += 1

    conn.commit()
    conn.close()

    print(f"Added {added_count} new files.")
    print(f"Skipped {skipped_count} duplicates.")

if __name__ == "__main__":
    scan_media_folder()