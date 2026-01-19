"""
Generate a simple label -> jenis mapping (organik/anorganik) file used by model.py.
You can run this and then edit the JSON to map specific subcategories to organik/anorganik.
"""

import json
from pathlib import Path
import os


def generate(data_dir, out='backend/label_to_jenis.json'):
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f'data_dir not found: {data_dir}')

    # --- Temukan semua folder kelas dari dataset ---
    classes = []
    for root, dirs, files in os.walk(p):
        imgs = [f for f in files if Path(f).suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if not imgs:
            continue
        rel = Path(root).relative_to(p)
        class_name = str(rel).replace(os.sep, '_') if str(rel) != '.' else p.name
        classes.append(class_name)
    classes = sorted(set(classes))

    mapping = {}

    # ✅ Kata kunci kategori organik
    organik_kw = [
        # label dataset kamu (pastikan sesuai CLASS_NAMES)
        'buah', 'bunga', 'campuran', 'daging', 'daun', 'makanan',

        # sinonim umum
        'nasi', 'fruit', 'apple', 'banana', 'orange', 'jeruk', 'apel', 'pisang',
        'sayur', 'vegetable', 'leaf', 'leaves', 'flower', 'mawar', 'kembang',
        'meat', 'chicken', 'beef', 'ayam', 'sapi', 'organic', 'organik', 'food'
    ]

    # ✅ Kata kunci kategori anorganik
    anorganik_kw = [
        'plastik', 'plastic', 'botol', 'bottle',
        'kertas', 'paper',
        'kardus', 'cardboard', 'box',
        'kaca', 'glass',
        'metal', 'logam', 'kaleng', 'can', 'aluminium'
    ]

    # --- Proses setiap kelas ---
    for c in classes:
        lc = c.lower()
        parts = lc.replace('-', '_').split('_')

        # ✅ Deteksi keyword organik/anorganik
        if any(k in lc for k in organik_kw) or any(p in organik_kw for p in parts):
            mapping[c] = 'organik'
        elif any(k in lc for k in anorganik_kw) or any(p in anorganik_kw for p in parts):
            mapping[c] = 'anorganik'
        else:
            # ✅ default ke organik (bukan anorganik) agar tidak salah klasifikasi
            mapping[c] = 'organik'

    # --- Simpan ke file JSON ---
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f'✅ Wrote label->jenis mapping to: {outp.resolve()}')
    print(json.dumps(mapping, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default='backend/label_to_jenis.json')
    args = parser.parse_args()
    generate(args.data_dir, out=args.out)
