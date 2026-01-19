Penjelasan Seluruh Kode — Proyek EcoClassify

1. Pendahuluan

Proyek EcoClassify merupakan sistem klasifikasi sampah berbasis machine learning yang dirancang untuk membantu pengenalan jenis sampah organik dan anorganik melalui citra visual. Sistem ini mengintegrasikan model pembelajaran mendalam (deep learning) berbasis EfficientNetB0 dengan antarmuka pengguna berbasis web.

Proyek ini memiliki tiga komponen utama:

Backend (server sisi Flask) — menangani prediksi, penyimpanan data riwayat klasifikasi, dan evaluasi model.

Frontend (aplikasi web) — menyediakan antarmuka bagi pengguna untuk mengunggah gambar dan menampilkan hasil klasifikasi.

Dataset — berisi kumpulan citra sampah yang dikelompokkan per kategori sebagai data latih dan uji.

Secara umum, sistem ini bekerja dengan mengunggah gambar melalui antarmuka web, yang kemudian dikirim ke server Flask untuk diproses menggunakan model klasifikasi. Hasil prediksi berupa kategori, jenis sampah, akurasi, serta saran edukatif dikembalikan ke pengguna dan dapat disimpan dalam riwayat klasifikasi.

2. Arsitektur Sistem

Struktur direktori proyek secara umum adalah sebagai berikut:

EcoClassify/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   ├── database.py
│   ├── evaluate_and_report.py
│   ├── generate_label_map.py
│   ├── edukasi.json
│   ├── label_to_jenis.json
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── Dataset/
│   └── Garbage classification/
│       ├── Organik/
│       └── Anorganik/
│
└── README.md


Komunikasi antara frontend dan backend dilakukan menggunakan permintaan HTTP (HTTP requests) dengan format JSON.

3. Penjelasan Tiap Komponen
3.1 Backend
a. app.py

File ini merupakan titik masuk utama (entry point) server Flask.

Fungsi utama:

Menginisialisasi aplikasi Flask dan database SQLite.

Menyediakan endpoint API berikut:

POST /api/predict → menerima gambar, menjalankan prediksi, dan mengembalikan hasil dalam format JSON.

GET /api/history → mengembalikan seluruh riwayat klasifikasi.

POST /api/history → menyimpan hasil prediksi ke database.

DELETE /api/history → menghapus seluruh data riwayat klasifikasi.

GET /api/statistics → mengembalikan statistik klasifikasi (total, jumlah organik/anorganik, akurasi rata-rata).

Alur umum fungsi /api/predict:

Menerima berkas gambar dari frontend.

Memverifikasi ekstensi file dengan fungsi allowed_file() dari utils.py.

Menyimpan gambar sementara di direktori tmp_upload/.

Memanggil predict_image() dari model.py.

Mengembalikan hasil prediksi dalam bentuk JSON.

Struktur output JSON:

{
  "kategori": "Buah",
  "jenis_sampah": "organik",
  "akurasi": 85.0,
  "edukasi": "Buah termasuk sampah organik. Dapat dikomposkan.",
  "confidence_ok": true
}

b. model.py

Modul ini berperan sebagai inti pemrosesan klasifikasi menggunakan model EfficientNetB0.

Fungsi utama:

load_model()
Memuat model terlatih dari file .keras atau .h5. Jika model tidak ditemukan, sistem akan menggunakan fallback heuristic berbasis nama file dan keyword matching.

predict_image(image_path)
Melakukan inferensi terhadap gambar yang diberikan dengan langkah:

Preprocessing: ubah gambar ke RGB, ubah ukuran menjadi 224×224 piksel, normalisasi nilai piksel.

Jalankan prediksi menggunakan model.

Hitung softmax untuk mendapatkan probabilitas.

Pilih label dengan probabilitas tertinggi.

Gunakan fungsi _map_to_jenis() untuk menentukan apakah label termasuk organik atau anorganik.

Jika tingkat kepercayaan < ambang batas (20%), maka hasil ditandai sebagai “tidak yakin”.

Ambil pesan edukatif melalui _get_edukasi().

_map_to_jenis(label)
Mengonversi nama label menjadi kategori jenis sampah berdasarkan file label_to_jenis.json. Jika tidak ditemukan, fungsi akan menggunakan daftar kata kunci organik/anorganik.

_get_edukasi(label, jenis)
Mengambil pesan edukasi yang sesuai dari file edukasi.json.

Catatan teknis:
Jika model tidak ditemukan, sistem tetap berjalan dengan fallback rules, yaitu klasifikasi berbasis nama file (contoh: file yang mengandung “buah” dikategorikan organik).

c. utils.py

Berisi fungsi bantu (utility functions) untuk keperluan umum.

Fungsi penting:

allowed_file(filename)
Memeriksa apakah ekstensi file termasuk yang diizinkan (png, jpg, jpeg, gif).

compute_statistics(rows)
Menghitung total klasifikasi, jumlah organik dan anorganik, serta akurasi rata-rata dari data yang disimpan dalam database.

d. database.py

Menangani operasi penyimpanan data riwayat menggunakan SQLite.

Fungsi utama:

init_db(path) → Membuat tabel database jika belum ada.

insert_history() → Menyimpan hasil klasifikasi baru.

get_all_history() → Mengambil seluruh riwayat.

clear_history() → Menghapus semua data riwayat.

Semua data disimpan dalam file history.db.

e. evaluate_and_report.py

Skrip untuk evaluasi batch model terhadap dataset.

Fungsi dan proses utama:

Menelusuri dataset dan membaca label kebenaran (ground truth).

Melakukan prediksi untuk setiap gambar melalui predict_image().

Menyimpan hasil per gambar ke per_image_predictions.csv.

Jika pustaka scikit-learn tersedia, menghitung precision, recall, F1-score, dan confusion matrix.

Menulis laporan ke eval_summary.json dan classification_report.txt.

f. generate_label_map.py

Skrip ini digunakan untuk membuat file label_to_jenis.json secara otomatis berdasarkan nama folder dataset.

Langkah kerja:

Membaca subfolder dalam dataset.

Menentukan apakah kelas tersebut organik atau anorganik menggunakan kata kunci.

Menulis hasil ke dalam file JSON.

3.2 Frontend

Frontend terletak pada folder frontend/ dan berisi file statis HTML, CSS, dan JavaScript.

a. index.html

Halaman utama antarmuka pengguna. Menyediakan elemen:

Form unggah gambar.

Tampilan hasil klasifikasi (kategori, jenis, akurasi, edukasi).

Navigasi ke halaman riwayat dan statistik.

b. app.js

File logika frontend utama yang menangani interaksi pengguna dan komunikasi dengan backend.

Fungsi utama:

predictImage(file) → Mengirim gambar ke endpoint /api/predict.

renderResult(json) → Menampilkan hasil prediksi di halaman.

saveResult() → Menyimpan hasil klasifikasi ke database.

loadHistory() dan loadStats() → Memuat data riwayat dan statistik dari backend.

Event handler untuk tombol navigasi dan unggah.

c. style.css

Mengatur tampilan visual aplikasi, termasuk tema terang/gelap dan tata letak komponen.

3.3 Dataset

Dataset terletak di folder Dataset/Garbage classification/ dan memiliki struktur seperti:

Dataset/Garbage classification/
├── Organik/
│   ├── buah/
│   ├── daun/
│   └── makanan/
└── Anorganik/
    ├── plastic/
    ├── glass/
    ├── metal/
    ├── paper/
    └── cardboard/


Masing-masing subfolder berisi kumpulan gambar sesuai kategorinya. Dataset ini digunakan untuk pelatihan, validasi, dan evaluasi model klasifikasi.

4. Alur Eksekusi Sistem

Alur eksekusi dari awal hingga akhir dapat dijelaskan sebagai berikut:

User (Frontend)
     │
     │ Upload gambar (HTTP POST)
     ▼
Flask API (/api/predict)
     │
     ├─> Simpan gambar sementara
     ├─> Panggil model.predict_image()
     │       ├─> Preprocess gambar
     │       ├─> Inference (EfficientNetB0)
     │       ├─> Hitung confidence
     │       ├─> Map ke jenis (organik/anorganik)
     │       └─> Ambil pesan edukasi
     │
     ▼
Kirim JSON hasil prediksi ke frontend
     │
     ▼
Frontend menampilkan kategori, akurasi, dan saran edukasi

5. Evaluasi Model

Evaluasi dilakukan menggunakan skrip evaluate_and_report.py. Skrip ini:

Menguji performa model terhadap seluruh dataset.

Menghasilkan metrik klasifikasi (precision, recall, F1-score).

Mengidentifikasi gambar yang diprediksi tidak yakin (unknown).

Menyimpan hasil ke folder backend/reports/.

6. Catatan Teknis dan Rekomendasi

Pastikan file model .keras atau .h5 tersedia di folder backend/.

Gunakan Python ≥ 3.9 dan pasang dependensi melalui requirements.txt.

Jalankan backend dengan perintah:

python backend/app.py


Buka frontend melalui browser (misalnya http://127.0.0.1:5000).

Jika model tidak ditemukan, sistem tetap berjalan dengan fallback heuristik.

Untuk evaluasi batch:

python backend/evaluate_and_report.py --data-dir "Dataset/Garbage classification" --out backend/reports

7. Kesimpulan

Proyek EcoClassify merepresentasikan integrasi menyeluruh antara model klasifikasi berbasis deep learning dan sistem antarmuka berbasis web yang ringan. Dengan pendekatan modular, sistem ini dapat dikembangkan lebih lanjut untuk mendukung klasifikasi multi-label, pengenalan otomatis komposisi sampah campuran, serta integrasi ke platform edukasi lingkungan.








ALGORITMA (singkat):

Saat menerima gambar:
Simpan sementara.
Pastikan model dimuat (load_model()).
Lakukan preprocessing -> inference -> softmax.
Ambil top label & prob.
Map label -> jenis (organik/anorganik) dengan label_to_jenis.json atau heuristik.
Jika prob < threshold -> tandai UNKNOWN / ‘tidak_yakin’.
Kembalikan hasil ke client.
FLOWCHART ASCII (end-to-end):

               +----------------+
               |   Browser UI   |
               | (upload image) |
               +--------+-------+
                        |
                        | POST /api/predict (image)
                        v
               +--------+--------+
               |   Flask app     |
               |  app.api_predict|
               +--------+--------+
                        |
                        | save temp image -> tmp_upload/
                        v
               +--------+--------+
               | backend.model   |
               | load_model()    |
               +--------+--------+
                        |
                 MODEL loaded? ----- no -----> use fallback_heuristic()
                        | yes
                        v
               +--------+--------+
               | predict_image() |
               | preprocess img  |
               | forward->softmax|
               | top_label, prob |
               +--------+--------+
                        |
                        v
               +--------+--------+
               | map label ->    |
               | jenis (json/heur)|
               +--------+--------+
                        |
                 prob >= THRESHOLD?
                  /            \
                yes             no
                 |               |
                 v               v
        confidence_ok=True   jenis='tidak_yakin'
                 |               |
                 v               v
            return JSON (kategori, jenis, akurasi, edukasi, confidence_ok)

