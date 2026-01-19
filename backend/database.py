"""
Simple SQLite helper for history table.
"""
import sqlite3
from datetime import datetime
from sqlite3 import Connection

CREATE_SQL = '''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY,
    nama_gambar TEXT,
    jenis_sampah TEXT,
    kategori_sampah TEXT,
    akurasi REAL,
    waktu DATETIME,
    saran_edukasi TEXT,
    confidence_ok INTEGER DEFAULT 1
);
'''

def get_conn(db_path: str) -> Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str):
    conn = get_conn(db_path)
    conn.execute(CREATE_SQL)
    conn.commit()
    # Migration: if existing table lacks confidence_ok column, add it
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(history)")
        cols = [r[1] for r in cur.fetchall()]
        if 'confidence_ok' not in cols:
            cur.execute("ALTER TABLE history ADD COLUMN confidence_ok INTEGER DEFAULT 1")
            conn.commit()
    except Exception:
        # best-effort migration; ignore if not possible
        pass
    conn.close()

def insert_history(db_path: str, payload: dict) -> int:
    conn = get_conn(db_path)
    cur = conn.cursor()
    waktu = payload.get('waktu') or datetime.now().isoformat()
    cur.execute(
        'INSERT INTO history (nama_gambar, jenis_sampah, kategori_sampah, akurasi, waktu, saran_edukasi, confidence_ok) VALUES (?,?,?,?,?,?,?)',
        (
            payload.get('nama_gambar'),
            payload.get('jenis_sampah'),
            payload.get('kategori_sampah') or payload.get('kategori'),
            payload.get('akurasi'),
            waktu,
            payload.get('saran_edukasi') or payload.get('edukasi'),
            1 if payload.get('confidence_ok', True) else 0
        )
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid

def get_all_history(db_path: str):
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute('SELECT * FROM history ORDER BY waktu DESC')
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def clear_history(db_path: str):
    conn = get_conn(db_path)
    conn.execute('DELETE FROM history')
    conn.commit()
    conn.close()


def delete_history_entry(db_path: str, row_id: int) -> bool:
    """Delete a single history row by id. Returns True if a row was deleted."""
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute('DELETE FROM history WHERE id = ?', (row_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
