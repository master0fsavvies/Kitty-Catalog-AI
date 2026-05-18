import sqlite3

DB_PATH = "catlib.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS media_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    media_type TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,

    caption TEXT DEFAULT '',
    tags TEXT DEFAULT '',
    vibe TEXT DEFAULT '',

    favorite INTEGER DEFAULT 0,
    date_added TEXT DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
conn.close()

print("Database and media_items table created.")