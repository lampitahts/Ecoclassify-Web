📘 README.md
🌿 EcoClassify — Smart Waste Classification

A web-based waste classification system using EfficientNetB0 deep learning model and Flask backend.

✨ Fitur Utama

🚀 Klasifikasi otomatis sampah organik dan anorganik dari gambar.

🧠 Model EfficientNetB0 dengan fallback heuristik jika model tidak tersedia.

💾 Penyimpanan riwayat klasifikasi berbasis SQLite.

📊 Statistik visual: jumlah klasifikasi, proporsi organik/anorganik, dan akurasi rata-rata.

🌍 Antarmuka web interaktif (HTML + JavaScript).

🧾 Evaluasi batch otomatis pada dataset untuk analisis performa.

🧩 Arsitektur Sistem
Frontend (HTML/JS)
      │
      ▼
Flask API (backend/app.py)
      │
      ▼
Model EfficientNetB0 → Prediksi
      │
      ▼
SQLite Database (history.db)

📂 Struktur Proyek
EcoClassify/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   ├── database.py
│   ├── evaluate_and_report.py
│   ├── label_to_jenis.json
│   ├── edukasi.json
│   ├── model_waste_classifier.keras
│   └── history.db
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
└── Dataset/
    └── Garbage classification/
        ├── Organik/
        └── Anorganik/

⚙️ Cara Menjalankan
1️⃣ Install dependensi
pip install -r backend/requirements.txt

2️⃣ Jalankan backend Flask
cd backend
python app.py

3️⃣ Jalankan frontend
cd frontend
python -m http.server 8000


Buka di browser:

http://127.0.0.1:8000

📸 Contoh Hasil Prediksi
Gambar	Prediksi	Jenis	Akurasi	Edukasi
🍌 buah.jpg	Buah	Organik	84.5%	Dapat dikomposkan atau jadi pakan ternak
🧃 plastic_bottle.png	Plastic	Anorganik	91.2%	Bersihkan dan pisahkan berdasarkan jenis plastik
🧪 Evaluasi Model

Jalankan:

python backend/evaluate_and_report.py \
  --data-dir "Dataset/Garbage classification" \
  --out backend/reports


Output:

eval_summary.json

per_image_predictions.csv

classification_report.txt

📘 File Penting
File	Fungsi
backend/app.py	API utama Flask
backend/model.py	Pemrosesan model dan prediksi
backend/database.py	Manajemen database riwayat
backend/utils.py	Fungsi bantu (validasi & statistik)
frontend/app.js	Logika frontend (upload, prediksi, render hasil)
backend/evaluate_and_report.py	Evaluasi batch model
backend/generate_label_map.py	Generator label → jenis sampah
🧠 Teknologi yang Digunakan

Python 3.10+

TensorFlow / Keras

Flask

SQLite

Chart.js (Frontend)

HTML, CSS, JavaScript

📜 Lisensi

Proyek ini bersifat open-source untuk keperluan akademik dan edukasi.
Lisensi: MIT License

🙌 Kontributor

👤 [Nama Kamu] — Pengembang utama backend & frontend.

🌱 Proyek ini dikembangkan sebagai bagian dari riset pengelolaan sampah cerdas berbasis AI.