import sqlite3


def init_db():
    conn = sqlite3.connect("tfidf.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            tf REAL NOT NULL,
            idf REAL NOT NULL,
            tf_idf REAL NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()
