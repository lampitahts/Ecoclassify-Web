"""
Simple Flask backend for EcoClassify.
Provides endpoints for prediction, saving history, fetching history and statistics.
Uses SQLite (database.py) and a placeholder model (model.py) that can be swapped with a trained ResNet.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from database import init_db, get_all_history, insert_history, clear_history, delete_history_entry
from model import load_model, predict_image
from utils import allowed_file
import os

app = Flask(__name__)
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')

# Initialize database on startup. Use the most compatible hook available
if hasattr(app, 'before_first_request'):
    @app.before_first_request
    def startup():
        # Initialize database only. Model loading is lazy to keep startup fast.
        init_db(DB_PATH)
elif hasattr(app, 'before_serving'):
    # Flask 2.0+/3.x may provide before_serving (async-capable)
    @app.before_serving
    async def startup():
        init_db(DB_PATH)
else:
    # Fallback: run immediately at import time
    init_db(DB_PATH)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400
    f = request.files['image']
    if f.filename == '' or not allowed_file(f.filename):
        return jsonify({'error': 'No selected file or file type not allowed'}), 400
    # Save temporarily
    tmp_path = os.path.join(os.path.dirname(__file__), 'tmp_upload')
    os.makedirs(tmp_path, exist_ok=True)
    save_path = os.path.join(tmp_path, f.filename)
    f.save(save_path)
    # Load model lazily (may be no-op if no trained model exists)
    try:
        load_model()
    except Exception:
        pass
    # Predict
    result = predict_image(save_path)
    # Do NOT auto-save prediction to history here.
    # Saving should only happen when frontend explicitly requests it via /api/history POST.
    # Return prediction result with confidence information only.
    result['saved'] = False
    result['history_id'] = None
    return jsonify(result)


@app.route('/api/history', methods=['GET', 'POST', 'DELETE'])
def api_history():
    if request.method == 'GET':
        # support filtering by jenis (organik/anorganik) and date range: start_date, end_date (ISO)
        jenis = request.args.get('jenis')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        rows = get_all_history(DB_PATH)
        # apply filters
        def in_range(r):
            if jenis and (r.get('jenis_sampah') != jenis and r.get('jenis_sampah') != ("organik" if jenis=="organik" else jenis)):
                return False
            if start_date:
                try:
                    from dateutil import parser
                    dt = parser.isoparse(r.get('waktu'))
                    if dt < parser.isoparse(start_date):
                        return False
                except Exception:
                    pass
            if end_date:
                try:
                    from dateutil import parser
                    dt = parser.isoparse(r.get('waktu'))
                    if dt > parser.isoparse(end_date):
                        return False
                except Exception:
                    pass
            return True

        rows = [r for r in rows if in_range(r)]

        # convert DB snake_case rows to frontend-friendly camelCase keys
        def to_camel(r):
            return {
                'id': r.get('id'),
                'namaGambar': r.get('nama_gambar') or r.get('namaGambar'),
                'waktu': r.get('waktu'),
                'kategori': r.get('kategori_sampah') or r.get('kategori') or r.get('kategori_sampah'),
                'jenisSampah': r.get('jenis_sampah') or r.get('jenisSampah'),
                'akurasi': r.get('akurasi'),
                'edukasi': r.get('saran_edukasi') or r.get('edukasi')
            }
        mapped = [to_camel(r) for r in rows]
        return jsonify({'history': mapped})
    if request.method == 'POST':
        payload = request.json or {}
        # normalize keys: accept camelCase from frontend or snake_case
        normalized = {
            'nama_gambar': payload.get('namaGambar') or payload.get('nama_gambar'),
            'kategori_sampah': payload.get('kategori') or payload.get('kategori_sampah') or payload.get('kategori_sampah'),
            'jenis_sampah': payload.get('jenisSampah') or payload.get('jenis_sampah'),
            'akurasi': payload.get('akurasi') or payload.get('akurasi'),
            'saran_edukasi': payload.get('edukasi') or payload.get('saran_edukasi')
        }
        row_id = insert_history(DB_PATH, normalized)
        return jsonify({'id': row_id})
    if request.method == 'DELETE':
        clear_history(DB_PATH)
        return jsonify({'cleared': True})


@app.route('/api/history/<int:row_id>', methods=['DELETE'])
def api_history_delete(row_id: int):
    """Delete a single history entry by id and return status."""
    try:
        deleted = delete_history_entry(DB_PATH, row_id)
        return jsonify({'deleted': bool(deleted)})
    except Exception:
        return jsonify({'deleted': False}), 500


@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    rows = get_all_history(DB_PATH)
    total = len(rows)
    if total == 0:
        return jsonify({
            'total': 0,
            'counts_per_category': {},
            'counts': {'organik': 0, 'anorganik': 0},
            'avg_accuracy': 0
        })

    # counts per category (kategori_sampah)
    from collections import Counter
    cat_counts = Counter([r.get('kategori_sampah') or r.get('kategori') for r in rows])
    jenis_counts = Counter([r.get('jenis_sampah') for r in rows])
    avg_accuracy = round(sum((r.get('akurasi') or 0) for r in rows) / total, 1)

    return jsonify({
        'total': total,
        'counts_per_category': dict(cat_counts),
        'counts': {'organik': jenis_counts.get('organik', 0), 'anorganik': jenis_counts.get('anorganik', 0)},
        'avg_accuracy': avg_accuracy
    })


# Serve frontend build index if present, otherwise provide a simple instruction page
@app.route('/')
def index():
    # Serve frontend files directly if frontend folder exists (development-friendly)
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    frontend_dir = os.path.join(base, 'frontend')
    # Prefer build outputs if present
    for d in (os.path.join(frontend_dir, 'dist'), os.path.join(frontend_dir, 'build')):
        index_path = os.path.join(d, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(d, 'index.html')
    # If raw frontend directory exists, serve its index.html
    index_path = os.path.join(frontend_dir, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(frontend_dir, 'index.html')
    # No frontend found — return helpful message
    msg = '''<!doctype html>
<html>
  <head><meta charset="utf-8"><title>EcoClassify API</title></head>
  <body>
    <h2>EcoClassify Backend</h2>
    <p>Server API berjalan. Endpoint tersedia di <code>/api/...</code></p>
    <p>Frontend tidak ditemukan. Pastikan folder <code>frontend/</code> ada atau build frontend ke <code>frontend/dist</code>.</p>
  </body>
</html>
'''
    return msg, 200, {'Content-Type': 'text/html'}



@app.route('/<path:filename>')
def static_proxy(filename):
    """Serve frontend static files (css, js, assets) from frontend/ when present.
    This keeps API under /api/* and everything else served by Flask for convenience in dev.
    """
    # Prevent overriding API routes
    if filename.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    frontend_dir = os.path.join(base, 'frontend')
    # check build folders first
    for d in (os.path.join(frontend_dir, 'dist'), os.path.join(frontend_dir, 'build')):
        file_path = os.path.join(d, filename)
        if os.path.exists(file_path):
            return send_from_directory(d, filename)
    # fallback to raw frontend folder
    file_path = os.path.join(frontend_dir, filename)
    if os.path.exists(file_path):
        return send_from_directory(frontend_dir, filename)
    # not found
    return ('', 404)


@app.route('/favicon.ico')
def favicon():
    # try to serve favicon from frontend build if present
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidates = [
        os.path.join(base, 'frontend', 'dist'),
        os.path.join(base, 'frontend', 'build'),
    ]
    for d in candidates:
        fav = os.path.join(d, 'favicon.ico')
        if os.path.exists(fav):
            return send_from_directory(d, 'favicon.ico')
    # otherwise return no content to avoid 404 log spam
    return ('', 204)


if __name__ == '__main__':
    # disable reloader for automated runs in this environment
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
