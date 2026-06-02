# Dokumentasi Teknis Kode Program
### Dashboard Interaktif & Model Deep Embedded Clustering (DEC)
Dokumentasi ini menjelaskan secara mendalam alur logika, fungsi, dan arsitektur kode pada berkas **`app.py`** dan notebook **`DEC_Best_Method_Colab.ipynb`** secara bertahap per blok kode dan sel jaringan.

---

## 1. STRUKTUR UTAMA APLIKASI WEB (`app.py`)

Aplikasi web dirancang menggunakan framework **Streamlit** dengan gaya antarmuka gelap terpadu untuk keperluan pengujian dan presentasi hasil analisis.

### Blok 1: Inisialisasi, Konfigurasi Halaman & Styling HSL (Baris 1 - 104)
*   **Fungsi**: 
    *   Mengatur parameter metadata halaman peramban Streamlit seperti judul dokumen, ikon, dan opsi tata letak lebar.
    *   Mengintegrasikan pustaka `pillow_heif` untuk memastikan kompatibilitas format gambar resolusi tinggi berbasis HEIC dari perangkat seluler.
    *   Menyuntikkan konfigurasi kustom CSS dengan skema warna gelap untuk memetakan visualisasi metrik utama secara terstruktur.

### Blok 2: Custom Layer `DECLayer` (Baris 126 - 153)
*   **Fungsi**: Kelas layer klaster kustom yang mewarisi arsitektur dasar `tf.keras.layers.Layer`.
    *   `build(input_shape)`: Menginisialisasi matriks pusat klaster (cluster centers) sebagai bobot teroptimasi yang dapat dilatih selama proses pemurnian spasial.
    *   `call(inputs)`: Menghitung jarak kuadrat Euclidean antara proyeksi dimensi laten masukan dengan pusat klaster terbobot. Jarak tersebut kemudian dikonversi menjadi keanggotaan probabilitas lunak (soft assignment) berdasarkan fungsi densitas Student-t dengan derajat kebebasan alpha = 1.0.
    *   `get_config()`: Mengembalikan metadata konfigurasi layer untuk menjamin fungsionalitas serialisasi dan deserialisasi model Keras berjalan dengan konsisten.

### Blok 3: Pemuatan Model & Rekonstruksi Jaringan Fungsional (`load_dec_models`) (Baris 154 - 203)
*   **Fungsi**: Prosedur pemuatan bobot latih model secara aman dari direktori penyimpanan penyimpanan lokal.
    *   **Metode**: Merekonstruksi grafik jaringan secara dinamis menggunakan API fungsional Keras dengan menarik layer terlatih EfficientNetV2-B2, Dense 512, Batch Normalization, Dense 256, dan Classifier 10D langsung dari model klasifikasi dasar (`finetuned_final.keras`).
    *   Menyusun kembali layer normalisasi L2 serta `DECLayer` di atas model fungsional, kemudian memuat bobot latih dari `dec_model.keras` menggunakan fungsi `.load_weights()`. Pendekatan ini memecahkan batasan penarikan otomatis bentuk output Keras pada layer Lambda kustom.

### Blok 4: Pemrosesan Gambar Input (`preprocess_for_prediction`) (Baris 204 - 217)
*   **Fungsi**: Standardisasi dimensi dan nilai piksel citra uji masukan agar selaras dengan tahapan pelatihan.
    *   Menerima berkas gambar masukan dan mengubah dimensinya secara spasial ke resolusi 224 x 224 piksel memanfaatkan teknik interpolasi Lanczos.
    *   Mengonversi matriks warna menjadi larik bertipe data float32 dengan rentang skala intensitas piksel asli [0, 255] guna mempertahankan distribusi fitur awal pada ekstraktor EfficientNetV2.

### Blok 5: Tab 1 — Real-Time Predictor & Manifest Gallery (Baris 257 - 381)
*   **Fungsi**: Mengontrol alur antarmuka pengujian citra real-time dan penyajian hasil klaster.
    *   Menyediakan komponen antarmuka untuk unggah gambar dan menampilkan perbandingan visual citra sebelum dan sesudah pembesaran skala.
    *   Mengeksekusi prediksi probabilitas keanggotaan klaster melalui model DEC serta mengekstrak proyeksi vektor spasial melalui model Encoder.
    *   Menampilkan nama klaster terprediksi beserta kategori nama museum hasil pemetaan 1-to-1, nilai persentase probabilitas, representasi grafik batang 10-dimensi, dan area visualisasi sebaran koordinat laten.
    *   Membaca struktur direktori lokal secara real-time untuk menyajikan galeri sampel gambar lain yang secara otomatis dikelompokkan ke dalam klaster yang sama.

### Blok 6: Tab 2 — Model Dashboard & Charts (Baris 382 - 440)
*   **Fungsi**: Panel visualisasi statistik hasil evaluasi model akhir.
    *   Menampilkan grafik komparasi evaluasi dari direktori penyimpanan:
        1.  `00_dec_training.png`: Progress konvergensi nilai loss KL-Divergence dan Silhouette Score selama iterasi pelatihan.
        2.  `02_confusion.png`: Heatmap visualisasi confusion matrix antara kelas target riil dengan klaster terprediksi.
        3.  `01_tsne_umap.png`: Peta visualisasi reduksi dimensi sebaran spasial 2D menggunakan algoritma t-SNE dan UMAP.
        4.  `03_silhouette.png`: Plot koefisien siluet per sampel citra untuk mengukur tingkat kerapatan dan keterpisahan klaster.

---

## 2. PIPELINE PELATIHAN MODEL (`DEC_Best_Method_Colab.ipynb`)

Notebook ini mencakup seluruh alur pemodelan terstruktur dari ekstraksi fitur spasial awal hingga proses optimasi berbasis KL-Divergence di Google Colab.

### Sel 1: Dokumentasi Deskripsi Pipeline (Markdown)
*   **Fungsi**: Memberikan penjelasan teoretis mengenai alur kerja jaringan dari ekstraksi EfficientNetV2-B2 hingga inisialisasi spasial *Identity Centroid* untuk mempertahankan korelasi 1-to-1 kelas museum.

### Sel 2: Google Colab Runtime Setup (Code)
*   **Fungsi**: 
    *   Menghubungkan direktori Google Drive eksternal ke lingkungan Google Colab menggunakan modul `drive.mount()`.
    *   Menginstal modul eksternal `pillow-heif` untuk memperluas pustaka pemrosesan gambar Apple HEIC.

### Sel 3: Pemuatan Pustaka & Penanganan Kepatuhan NumPy (Code)
*   **Fungsi**:
    *   Melakukan pembaruan tipe data konstan NumPy agar kompatibel dengan fungsionalitas visualisasi TensorFlow versi terbaru.
    *   Mengimpor pustaka analisis matematika dasar, visualisasi data, pengklusteran Scikit-Learn, reduksi dimensi, dan framework deep learning TensorFlow.
    *   Memastikan penguncian seed acak di angka 42 untuk menjaga konsistensi hasil pengujian.

### Sel 4: Konfigurasi Parameter Global (`CONFIG`) (Code)
*   **Fungsi**: Menginisialisasi kamus parameter kontrol utama, meliputi dimensi laten, jumlah klaster, learning rate, loss weight, dan otomatisasi pembuatan subdirektori output hasil latihan di Google Drive.

### Sel 5: Data Pipeline & Image Loader (`load_image_rgb`) (Code)
*   **Fungsi**: 
    *   Mendefinisikan fungsi pembacaan citra masukan RGB untuk berbagai format ekstensi.
    *   Memindai subdirektori kategori secara rekursif, melakukan encoding label secara terurut, dan memuat citra ke dalam larik memori berskala float32 [0, 255] dengan dimensi spasial 224 x 224 piksel.

### Sel 6: Definisi `DECLayer` Kustom (Code)
*   **Fungsi**: Menyusun kelas algoritma pembobotan spasial berbasis kedekatan Student-t distribution untuk menghitung soft assignment pada klaster.

### Sel 7: Pemuatan Model Fine-Tuned (Backbone) (Code)
*   **Fungsi**: Memuat arsitektur dasar model klasifikasi terlatih (`finetuned_final.keras`) dan membekukan seluruh parameter bobot pada ekstraktor EfficientNetV2-B2.

### Sel 8: Rekonstruksi Shared Projection Head & Model Komplit (Code)
*   **Fungsi**: 
    *   Menyambungkan kembali seluruh layer Dense terlatih dan Batch Normalization langsung ke bentuk grafik fungsional Keras yang baru.
    *   Mendefinisikan model fungsional DEC, model Encoder laten, serta model sub-kepala cepat (fast training sub-models) berbasis input dimensi 1408D.

### Sel 9: Pra-Ekstraksi Fitur Backbone (EfficientNetV2) (Code)
*   **Fungsi**: Mengekstrak representasi fitur citra masukan menggunakan backbone terbeku secara satu kali untuk mempercepat waktu pemrosesan iterasi pelatihan DEC.

### Sel 10: Inisialisasi Identity Centroid (Direct Class Mapping) (Code)
*   **Fungsi**: Menginisialisasi pusat klaster DEC menggunakan Matriks Identitas 10 x 10 guna menjamin pemetaan terarah langsung 1-to-1 antar kelas museum sejak awal iterasi.

### Sel 11: Setup Fungsi Loss KL-Divergence & Optimizer (Code)
*   **Fungsi**: Mendefinisikan perhitungan matematis KL-Divergence, target distribution P, dan mengonfigurasi Adam Optimizer.

### Sel 12: Loop Pelatihan Iteratif DEC (Joint Loss Optimization) (Code)
*   **Fungsi**: Mengoptimasi parameter bobot spasial melalui kalkulasi Joint Loss (KL-Loss + lambda * Cross-Entropy Loss) dan melakukan kriteria penghentian dini (early stopping) berdasarkan ambang batas toleransi delta prediksi.

### Sel 13: Evaluasi Akhir & Pemetaan Spasial 1-to-1 (Code)
*   **Fungsi**: 
    *   Mengevaluasi kinerja model final dan menyajikan perbandingan statistik Silhouette Score & Davies-Bouldin Index sebelum dan sesudah optimasi DEC.
    *   Menghitung akurasi Hungarian serta menyimpan representasi matriks prediksi laten secara permanen ke Google Drive.

### Sel 14: Pengeplotan & Penyimpanan Grafik Reduksi Dimensi 2D (Code)
*   **Fungsi**: Memproyeksikan sebaran vektor laten 10D ke dalam koordinat 2D menggunakan t-SNE dan UMAP, lalu menyimpan grafik visualisasinya ke disk.

### Sel 15: Penyusunan Heatmap Confusion Matrix (Code)
*   **Fungsi**: Membuat heatmap matriks kebingungan antara kategori data aktual aktual dengan prediksi pengelompokan DEC.

### Sel 16: Visualisasi Analisis Silhouette Per-Sampel (Code)
*   **Fungsi**: Menyusun sebaran visual koefisien siluet per sampel citra untuk masing-masing klaster terpisah.
