from typing import List
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_statistics(rows: List[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            'totalKlasifikasi': 0,
            'sampahOrganik': 0,
            'sampahAnorganik': 0,
            'akurasiRataRata': 0
        }
    sampahOrganik = sum(1 for r in rows if r.get('jenis_sampah') == 'organik')
    sampahAnorganik = sum(1 for r in rows if r.get('jenis_sampah') == 'anorganik')
    akurasiRataRata = round(sum((r.get('akurasi') or 0) for r in rows) / total, 1)
    return {
        'totalKlasifikasi': total,
        'sampahOrganik': sampahOrganik,
        'sampahAnorganik': sampahAnorganik,
        'akurasiRataRata': akurasiRataRata
    }
