

## **Link Project OneDrive**

[https://mikroskilacid-my.sharepoint.com/:f:/g/personal/221110776_students_mikroskil_ac_id/Eiey9HEy7PZAv006Y19rs-8BGM6IoH5dmimGVU-Yi50rHA?e=XetypI](https://mikroskilacid-my.sharepoint.com/:f:/g/personal/221110776_students_mikroskil_ac_id/Eiey9HEy7PZAv006Y19rs-8BGM6IoH5dmimGVU-Yi50rHA?e=XetypI)

## **Kelompok 12**

* 221113096 – Lampita E. R. Hutasoit
* 221110776 – Xavier William Kurniawan
* 221112888 – Steven


---

## **Deskripsi Proyek: EcoClassify**

Proyek **EcoClassify** merupakan sistem klasifikasi sampah berbasis *machine learning* yang dirancang untuk membantu pengenalan jenis sampah organik dan anorganik melalui citra visual. Sistem ini mengintegrasikan model pembelajaran mendalam (*deep learning*) berbasis **EfficientNetB0** dengan antarmuka pengguna berbasis web.

Proyek ini terdiri dari tiga komponen utama:

1. **Backend (Flask Server)**
   Menangani proses prediksi, penyimpanan data riwayat klasifikasi, dan evaluasi model.

2. **Frontend (Aplikasi Web)**
   Menyediakan antarmuka pengguna untuk mengunggah gambar dan menampilkan hasil klasifikasi.

3. **Dataset**
   Berisi kumpulan citra sampah yang dikelompokkan per kategori sebagai data latih dan uji.

Secara umum, sistem bekerja dengan cara pengguna mengunggah gambar melalui antarmuka web. Gambar tersebut dikirim ke server Flask untuk diproses menggunakan model klasifikasi. Hasil prediksi berupa kategori, jenis sampah, akurasi, serta saran edukatif dikembalikan ke pengguna dan dapat disimpan dalam riwayat klasifikasi.

---

## **Langkah Menjalankan Proyek EcoClassify**

### 1. Persiapan Lingkungan

a. **Membuat environment Python (disarankan)**

```
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

b. **Instalasi dependensi**
Masuk ke folder backend, lalu jalankan:

```
pip install -r requirements.txt
```

---

### 2. Struktur Direktori

Pastikan struktur proyek sesuai dengan format berikut:

```
EcoClassify/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   ├── database.py
│   ├── eval_summary.json
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
```

---

### 3. Menjalankan Server Backend (Flask)

Masuk ke folder backend dan jalankan:

```
cd backend
python app.py
```

Server akan berjalan di:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Pastikan variabel `apiBase` di file **frontend/app.js** menunjuk ke alamat di atas:

```javascript
const apiBase = 'http://127.0.0.1:5000';
```

---

### 4. Menjalankan Frontend

**Opsi 1 — Jalankan langsung dari file:**
Buka `frontend/index.html` di browser (disarankan menggunakan Chrome atau Edge).

**Opsi 2 — Jalankan melalui server lokal:**

```
cd frontend
python -m http.server 8000
```

Kemudian buka di browser:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### 5. Melakukan Prediksi

1. Pilih gambar sampah pada halaman utama.
2. Klik tombol **“Klasifikasikan”**.
3. Hasil yang ditampilkan mencakup:

   * **Kategori**: label prediksi (contoh: Buah, Plastik, dll.)
   * **Jenis**: organik atau anorganik
   * **Akurasi**: tingkat keyakinan model
   * **Edukasi**: saran pengelolaan sampah
4. Tekan **“Simpan ke Riwayat”** untuk menyimpan hasil ke database.

---

### 6. Melihat Riwayat dan Statistik

Navigasi ke tab **“Riwayat”** untuk melihat daftar klasifikasi yang telah disimpan.
Tab **“Statistik”** menampilkan:

* Jumlah total klasifikasi
* Distribusi organik vs anorganik
* Akurasi rata-rata

Semua data diambil dari **database backend/history.db**.

---

### 7. Evaluasi Batch (Opsional)

Untuk menilai performa model pada seluruh dataset, jalankan perintah berikut:

```
python backend/evaluate_and_report.py \
  --data-dir "Dataset/Garbage classification" \
  --out backend/reports \
  --threshold 40.0
```

**Output yang dihasilkan:**

* `per_image_predictions.csv` — hasil prediksi per gambar
* `eval_summary.json` — ringkasan metrik klasifikasi
* `classification_report.txt` — laporan hasil klasifikasi
* `confusion_matrix_all.png` — visualisasi matriks kebingungan (jika `matplotlib` tersedia)

---

### 8. Regenerasi File Label Mapping (Jika Dataset Diubah)

Jika ada kelas baru yang ditambahkan ke dataset, jalankan perintah berikut:

```
python backend/generate_label_map.py \
  --data_dir "Dataset/" \
  --out backend/label_to_jenis.json
```

---

### 9. Tips dan Troubleshooting

| Permasalahan                      | Penyebab Umum                   | Solusi                                                                               |
| --------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| Model tidak yakin (“tidak_yakin”) | Confidence rendah               | Turunkan nilai `ECOCONF_THRESHOLD` di environment, misalnya `ECOCONF_THRESHOLD=15.0` |
| Model tidak ditemukan             | File `.keras` atau `.h5` hilang | Pastikan file model diletakkan di folder `backend/`                                  |
| Riwayat tidak tersimpan           | Database belum terbentuk        | Jalankan ulang `app.py` agar `history.db` dibuat otomatis                            |
| Gagal memuat dataset              | Struktur folder tidak sesuai    | Pastikan subfolder mengikuti pola kelas per kategori                                 |
