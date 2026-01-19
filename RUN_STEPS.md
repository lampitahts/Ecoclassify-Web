🧭 RUN_STEPS.md
🚀 Langkah Menjalankan Proyek EcoClassify

Dokumen ini menjelaskan cara lengkap menjalankan proyek EcoClassify — dari instalasi, konfigurasi model, hingga menjalankan antarmuka web dan evaluasi batch.

1️⃣ Persiapan Lingkungan
a. Buat environment Python (disarankan)
python -m venv venv
source venv/bin/activate      # (Linux / macOS)
venv\Scripts\activate         # (Windows)

b. Install dependensi

Masuk ke folder backend/ lalu jalankan:

pip install -r requirements.txt

2️⃣ Struktur Direktori

Pastikan struktur proyekmu seperti berikut:

EcoClassify/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   ├── database.py
│   ├── evaluate_and_report.py
│   ├── generate_label_map.py
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
        │   ├── buah/
        │   ├── daun/
        │   └── makanan/
        └── Anorganik/
            ├── plastic/
            ├── paper/
            ├── glass/
            ├── metal/
            └── cardboard/

3️⃣ Menjalankan Server Backend (Flask)

Masuk ke folder backend:

cd backend
python app.py


Server akan berjalan di:

http://127.0.0.1:5000


Jika kamu menggunakan frontend statis (HTML/JS), pastikan apiBase di frontend/app.js menunjuk ke alamat di atas:

const apiBase = 'http://127.0.0.1:5000';

4️⃣ Menjalankan Frontend
Opsi 1 — Jalankan langsung dari file

Buka frontend/index.html di browser.
(Gunakan Chrome / Edge untuk dukungan JavaScript penuh.)

Opsi 2 — Jalankan via server lokal
cd frontend
python -m http.server 8000


Lalu buka di browser:

http://127.0.0.1:8000

5️⃣ Melakukan Prediksi

Pilih gambar sampah pada halaman utama.

Klik tombol “Klasifikasikan”.

Hasil ditampilkan dengan informasi:

Kategori: label prediksi (misal: Buah, Plastik, dll.)

Jenis: organik / anorganik

Akurasi: tingkat keyakinan model

Edukasi: saran pengelolaan sampah

Tekan “Simpan ke Riwayat” jika ingin menyimpannya ke database.

6️⃣ Melihat Riwayat dan Statistik

Navigasi ke tab “Riwayat” untuk melihat daftar klasifikasi yang disimpan.

Tab “Statistik” menampilkan jumlah total klasifikasi, distribusi organik vs anorganik, dan akurasi rata-rata.

Semua data diambil dari database backend/history.db.

7️⃣ Evaluasi Batch (Opsional)

Untuk menilai performa model pada seluruh dataset:

python backend/evaluate_and_report.py \
  --data-dir "Dataset/Garbage classification" \
  --out backend/reports \
  --threshold 40.0


Output:

📄 per_image_predictions.csv — hasil prediksi setiap gambar

📊 eval_summary.json — ringkasan metrik klasifikasi

📈 classification_report.txt — laporan tekstual

🧩 confusion_matrix_all.png — (jika matplotlib tersedia)

8️⃣ Regenerasi File Label Mapping (Jika Dataset Diubah)

Jika menambah kelas baru dalam dataset, jalankan:

python backend/generate_label_map.py \
  --data_dir "Dataset/Garbage classification" \
  --out backend/label_to_jenis.json

9️⃣ Tips & Troubleshooting
Permasalahan	Penyebab Umum	Solusi
Model tidak yakin (“tidak_yakin”)	Confidence rendah	Turunkan ECOCONF_THRESHOLD di environment, misal set ECOCONF_THRESHOLD=15.0
Model tidak ditemukan	File .keras/.h5 hilang	Letakkan file model di folder backend/
Riwayat tidak tersimpan	Database belum terbentuk	Jalankan ulang app.py agar history.db dibuat otomatis
Gagal memuat dataset	Struktur folder tidak sesuai	Pastikan subfolder mengikuti pola kelas per kategori