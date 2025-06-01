# Laporan Proyek Machine Learning - Sistem Rekomendasi Webtoon

## Project Overview

### Latar Belakang
Webtoon telah menjadi salah satu bentuk hiburan digital yang sangat populer di seluruh dunia. Dengan banyaknya judul yang tersedia di platform, pengguna sering kesulitan menemukan webtoon yang sesuai dengan preferensi mereka. Sistem rekomendasi dapat membantu pengguna menemukan webtoon baru berdasarkan genre, rating, dan preferensi pembaca lain, sehingga meningkatkan engagement dan kepuasan pengguna.[WebtoonNaver](https://help.naver.com/service/9732/contents/3325?lang=id&osType=COMMONOS)

### Mengapa Proyek Ini Penting?
1. **Meningkatkan User Experience**: Memudahkan pengguna menemukan konten yang sesuai dengan preferensi mereka
2. **Diversifikasi Konten**: Memperkenalkan pengguna pada genre atau karya yang mungkin belum mereka kenal
3. **Meningkatkan Engagement**: Mendorong pengguna untuk menghabiskan lebih banyak waktu di platform
4. **Membantu Kreator**: Meningkatkan peluang karya-karya berkualitas namun kurang populer untuk ditemukan

## Business Understanding

### Problem Statements
1. Bagaimana cara memberikan rekomendasi webtoon yang relevan kepada pengguna berdasarkan konten yang telah mereka baca atau sukai sebelumnya?
2. Bagaimana mengatasi cold-start problem ketika pengguna baru belum memiliki riwayat pembacaan atau rating yang cukup untuk dijadikan dasar rekomendasi?
3. Algoritma mana yang lebih efektif untuk dataset webtoon ini: content-based filtering, collaborative filtering, atau gabungan dari keduanya?

### Goals
1. Mengembangkan sistem rekomendasi dengan pendekatan content-based filtering yang dapat merekomendasikan webtoon berdasarkan kemiripan genre, cerita, dan karakteristik konten.
2. Mengembangkan sistem rekomendasi dengan pendekatan collaborative filtering yang dapat memberikan rekomendasi berdasarkan pola rating dan preferensi pengguna lain.
3. Membandingkan dan mengevaluasi performa kedua pendekatan untuk menentukan metode yang paling efektif untuk dataset webtoon.
4. Mengusulkan pendekatan hybrid yang menggabungkan kelebihan dari kedua metode untuk mengatasi keterbatasan masing-masing.

### Solution Approach
#### 1. **Content-Based Filtering dengan TF-IDF dan Cosine Similarity**
- **Teknik**: Menggunakan TF-IDF Vectorization untuk mengekstrak fitur dari konten webtoon
- **Algoritma**: Cosine Similarity untuk mengukur kesamaan antar webtoon
- **Target Metrik**: 
  - Diversitas genre ≥ 5 genre per rekomendasi
  - Coverage genre ≥ 75%
  - Rating rata-rata rekomendasi ≥ 9.0
- **Keunggulan**: Mengatasi cold-start problem dan memberikan rekomendasi berdasarkan konten

#### 2. **Collaborative Filtering dengan Matrix Factorization**
- **Teknik**: Non-negative Matrix Factorization (NMF) untuk dekomposisi matriks user-item
- **Algoritma**: Prediksi rating berdasarkan faktor laten pengguna dan item
- **Target Metrik**:
  - RMSE ≤ 1.5
  - MAE ≤ 1.0
  - Precision ≥ 0.7 (threshold rating = 7.0)
  - F1-Score ≥ 0.75
- **Keunggulan**: Menangkap pola preferensi komunitas dan memberikan rekomendasi personal

## Data Understanding

### Sumber Data
Dataset yang digunakan dalam proyek ini adalah Webtoon Comics Dataset yang tersedia di [Kaggle](https://www.kaggle.com/datasets/swarnimrai/webtoon-comics-dataset). Dataset ini berisi informasi tentang webtoon populer dari platform LINE Webtoon.


### Informasi Dataset
- Jumlah data: 569 Baris
- Kondisi data: missing values 1 pada subscribers_numeric
- Fitur yang tersedia: id, name, writer, likes, genre, rating, subscribers,summary, update, reading link

### Variabel/Fitur
Dataset ini memiliki beberapa variabel penting yang dapat digunakan untuk sistem rekomendasi:

1. **id**: Nomor identifikasi unik untuk setiap webtoon
2. **Name**: Judul webtoon
3. **Writer**: Penulis atau kreator webtoon
4. **Likes**: Jumlah likes yang diberikan oleh pengguna (dalam format angka dengan M untuk juta atau K untuk ribu)
5. **Genre**: Kategori atau genre webtoon (Romance, Action, Drama, dll)
6. **Rating**: Rating rata-rata dari pengguna (skala 1-10)
7. **Subscribers**: Jumlah pelanggan atau pengikut webtoon (dalam format angka dengan M untuk juta atau K untuk ribu)
8. **Summary**: Ringkasan atau sinopsis cerita
9. **Update**: Jadwal update webtoon
10. **Reading Link**: URL untuk membaca webtoon

### Exploratory Data Analysis

#### 2.1 Kualitas Data
- **Missing Values**: 1 missing value pada kolom Writer dan 1 pada Subscribers_Numeric
- **Data Duplikat**: Tidak ditemukan data duplikat
- **Format Data**: Kolom Likes dan Subscribers perlu preprocessing untuk konversi numerik

#### 2.2 Distribusi Genre
- **Genre Terpopuler**: Fantasy mendominasi dataset dengan 95 webtoon
- **Variasi Genre**: Terdapat 16 genre unik dalam dataset
- **Genre Berdasarkan Popularitas**: Romance memiliki rata-rata subscribers tertinggi (884,738 subscribers)

#### 2.3 Analisis Rating
- **Rating Tertinggi**: 9.93
- **Rating Terendah**: 5.41
- **Rating Rata-rata**: 9.42

#### 2.4 Analisis Popularitas
- **Webtoon Paling Populer (Likes)**: "My Giant Nerd Boyfriend" dengan 50.6M likes
- **Webtoon Paling Populer (Subscribers)**: "True Beauty" dengan 6.4M subscribers
- **Korelasi Likes vs Subscribers**: 0.85 (korelasi yang sangat kuat)
- **Korelasi Rating vs Subscribers**: 0.27 (korelasi lemah)


## Data Preparation

### Teknik yang Digunakan

1. **Data Cleaning** - Untuk mengatasi nilai kosong (missing values) dan memastikan kosnistensi data
2. **Feature Engineering** - Menciptakan representasi yang kaya untuk konten webtoon
3. **Text Preprocessing** - Menormalisasi dan membersihkan teks untuk analisis yang lebih baik
4. **Writer Style Profiling** - Mengidentifikasi pola genre dari penulis untuk memperkaya fitur
5. **TF-IDF Vectorization** - Mengubah fitur teks menjadi representasi vektor numerik
6. **Similarity Matrix Calculation** - Mengukur kesamaan antar item untuk rekomendasi


### Proses Preparation
1. **Data Cleaning**
- Penanganan Missing Values:
    - Kolom Writer diisi dengan 'Unknown Writer'
    - Kolom Genre diisi dengan 'Uncategorized'
    - Kolom Subscribers_Numeric diisi dengan 0
-  Standarisasi Format Numerik:
    - Konversi nilai Likes dan Subscribers dari format string ('M', 'K') ke nilai numerik (misal: "5.4M" → 5,400,000)

2. **Text Preprocessing**
df['Summary_Clean'] = df['Summary'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', ' ', x.lower()))

3. **Feature Engineering** 

df['Content_Features'] = df['Genre'] + ' ' + df['Writer'] + ' ' + df['Summary_Clean']

4. Vectorization dan Similarity Matrix
- **TF-IDF Matrix Shape**: (569, 6118) - 569 webtoon dengan 6118 unique terms
- **Cosine Similarity Matrix**: (569, 569) untuk mengukur kesamaan antar webtoon

5. Simulasi Data User-Item
- **Jumlah Pengguna**: 50 pengguna simulasi
- **Total Rating**: 979 rating dari 50 pengguna untuk 569 webtoon
- **User-Item Matrix Shape**: (50, 466)
- **Train-Test Split**: 783 training data, 196 testing data


writer_genre_mapping = df.groupby('Writer')['Genre'].agg(lambda x: ' '.join(x)).to_dict()
df['Writer_Style'] = df['Writer'].map(writer_genre_mapping)

## Modeling and Result

### Content-Based Filtering
Content-Based Filtering

**Implementasi:**

- Algoritma: TF-IDF Vectorization + Cosine Similarity
- Input: Content_Features (gabungan Genre + Writer + Summary_Clean)
- Output: 10 rekomendasi webtoon dengan similarity score tertinggi

**Cara Kerja:**

1. Mencari indeks webtoon berdasarkan judul
2. Mengambil skor similarity dari cosine_sim matrix
3. Mengurutkan berdasarkan skor tertinggi (tidak termasuk diri sendiri)
4. Mengambil 10 webtoon teratas

**Hasil Rekomendasi untuk "Tower of God":**

Rekomendasi teratas: "Tales of Greed" (Genre: Thriller, Similarity Score: 0.165)
Mayoritas rekomendasi dari genre Thriller, Action, dan Fantasy
Rata-rata rating rekomendasi: 9.35
Hasil Rekomendasi untuk "Omniscient Reader":

Rekomendasi teratas: "The First Night With the Duke" (Genre: Fantasy, Similarity Score: 0.156)
Mayoritas rekomendasi dari genre Fantasy dan Romance
Konten yang direkomendasikan memiliki tema fantasi serupa


### Collaborative Filtering

**Implementasi:**

- Algoritma: Non-negative Matrix Factorization (NMF)
- Faktor Laten: 10 dimensi
- Error Rekonstruksi: 246.1684


**Cara Kerja:**

1. Matriks user-item yang sparse diisi dengan nilai 0
2. NMF memfaktorisasi matriks menjadi dua matriks (W: user factors, H: item factors)
3. Rekonstruksi matriks lengkap dengan dot product W dan H
Rekomendasi didasarkan pada prediksi rating tertinggi untuk item yang belum diberi rating
Hasil Rekomendasi untuk User ID 5:

Rekomendasi teratas: "Daily JoJo" (Genre: Romance, Rating Prediksi: 5.85)
Variasi genre yang direkomendasikan: Romance, Fantasy, Drama, Action
Webtoon dengan rating asli yang tinggi (9.06 - 9.80)
Performa Model:

RMSE: 6.9733
MAE: 6.6389
Akurasi: 93.03%
76.22% prediksi berada dalam jarak 1 poin dari rating aktual
89.68% prediksi berada dalam jarak 2 poin dari rating aktual


### Kelebihan dan Kekurangan

#### Content-Based Filtering:
- **Kelebihan**:
  - **No Cold-Start Problem**: Dapat memberikan rekomendasi untuk item baru tanpa memerlukan data interaksi pengguna
  - **Transparansi**: Rekomendasi dapat dijelaskan berdasarkan fitur konten yang mirip
  - **User Independence**: Tidak bergantung pada data pengguna lain, sehingga privasi lebih terjaga
  - **Domain Knowledge Integration**: Dapat memanfaatkan pengetahuan domain (genre, penulis) secara efektif
  - **Konsistensi Temporal**: Preferensi konten cenderung stabil dari waktu ke waktu
  - **Hasil Penelitian**: Diversitas genre tinggi (6.10/rekomendasi) dan coverage luas (81.25%)

- **Kekurangan**:
  - **Limited Content Analysis**: Hanya bergantung pada fitur yang dapat diekstrak dari konten
  - **Over-specialization**: Cenderung merekomendasikan item yang terlalu mirip (filter bubble)
  - **Feature Engineering Dependency**: Kualitas sangat bergantung pada representasi fitur yang dibuat
  - **Lack of Serendipity**: Sulit memberikan rekomendasi yang mengejutkan namun relevan
  - **Scalability Issues**: Pemrosesan fitur teks dapat menjadi bottleneck pada dataset besar

#### Collaborative Filtering:
- **Kelebihan**:
  - **Community Wisdom**: Memanfaatkan pola kolektif dari komunitas pengguna
  - **Serendipity**: Dapat menemukan item yang tidak terduga namun relevan
  - **No Domain Knowledge Required**: Tidak memerlukan pemahaman mendalam tentang domain
  - **Implicit Feedback**: Dapat bekerja dengan data rating implisit dan eksplisit
  - **Social Aspect**: Menangkap tren dan preferensi sosial
  - **Hasil Penelitian**: Coverage sempurna (100% user dan item), Precision tinggi (79.82%)

- **Kekurangan**:
  - **Cold-Start Problem**: Kesulitan menangani pengguna atau item baru
  - **Data Sparsity**: Memerlukan jumlah rating yang cukup untuk performa optimal
  - **Popularity Bias**: Cenderung merekomendasikan item populer
  - **Scalability Challenges**: Kompleksitas komputasi meningkat dengan ukuran dataset
  - **Black Box**: Sulit menjelaskan mengapa suatu item direkomendasikan
  - **Hasil Penelitian**: RMSE tinggi (6.97) menunjukkan ketidakakuratan prediksi rating absolut

## Evaluation

### Metrik Evaluasi

#### Content-Based Filtering:
1. **Diversitas Genre**: Rata-rata jumlah genre unik dalam rekomendasi
   - **Formula**: `Diversitas = (1/N) × Σᵢ|Gᵢ|`
   - **Penjelasan**: N = jumlah rekomendasi, |Gᵢ| = jumlah genre unik dalam rekomendasi ke-i
   - **Interpretasi**: Semakin tinggi, semakin beragam genre yang direkomendasikan
   
2. **Cakupan Genre**: Persentase genre yang tercakup dalam rekomendasi dari total genre
   - **Formula**: `Coverage = (|G_rec| / |G_total|) × 100%`
   - **Penjelasan**: G_rec = genre yang muncul dalam rekomendasi, G_total = semua genre dalam dataset
   - **Interpretasi**: Nilai 100% berarti semua genre terwakili dalam rekomendasi
   
3. **Cosine Similarity**: Mengukur kesamaan antar vektor konten
   - **Formula**: `sim(A,B) = (A·B) / (||A|| × ||B||)`
   - **Penjelasan**: A·B = dot product, ||A|| = magnitude vektor A
   - **Interpretasi**: Nilai 0-1, dimana 1 = identik, 0 = tidak ada kesamaan

#### Collaborative Filtering:
1. **RMSE (Root Mean Square Error)**: Mengukur akar dari rata-rata kuadrat error prediksi
   - **Formula**: `RMSE = √[(1/n) × Σᵢ(rᵢ - r̂ᵢ)²]`
   - **Penjelasan**: rᵢ = rating aktual, r̂ᵢ = rating prediksi, n = jumlah prediksi
   - **Interpretasi**: Semakin rendah semakin baik, memberikan penalti lebih besar pada error besar
   
2. **MAE (Mean Absolute Error)**: Mengukur rata-rata nilai absolut error
   - **Formula**: `MAE = (1/n) × Σᵢ|rᵢ - r̂ᵢ|`
   - **Penjelasan**: Rata-rata dari selisih absolut antara prediksi dan aktual
   - **Interpretasi**: Lebih robust terhadap outlier dibanding RMSE
   
3. **Precision**: Proporsi rekomendasi relevan dari semua rekomendasi
   - **Formula**: `Precision = TP / (TP + FP)`
   - **Penjelasan**: TP = True Positive, FP = False Positive (threshold rating = 7.0)
   - **Interpretasi**: Mengukur akurasi rekomendasi positif
   
4. **Recall**: Proporsi item relevan yang berhasil direkomendasikan
   - **Formula**: `Recall = TP / (TP + FN)`
   - **Penjelasan**: FN = False Negative, mengukur kelengkapan deteksi item relevan
   - **Interpretasi**: Tinggi berarti sedikit item relevan yang terlewat
   
5. **F1-Score**: Harmonic mean dari precision dan recall
   - **Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
   - **Penjelasan**: Menyeimbangkan trade-off antara precision dan recall
   - **Interpretasi**: Metrik tunggal yang menggabungkan kedua aspek akurasi

#### Content-Based Filtering:
1. **Diversitas Genre**: Rata-rata jumlah genre unik dalam rekomendasi
   - Formula: `Avg(count(unique genres per recommendation))`
   
2. **Cakupan Genre**: Persentase genre yang tercakup dalam rekomendasi dari total genre
   - Formula: `(jumlah_genre_dalam_rekomendasi / jumlah_total_genre) × 100%`
   
3. **Kesamaan Konten**: Rata-rata skor similarity antar webtoon yang direkomendasikan
   - Formula: `Avg(similarity_scores)`
   
4. **Kualitas Rekomendasi**: Rating rata-rata dan standar deviasi webtoon yang direkomendasikan
   - Formula: `Avg(ratings)`, `Std(ratings)`

#### Collaborative Filtering:
1. **RMSE (Root Mean Square Error)**: Mengukur akar dari rata-rata kuadrat error prediksi
   - Formula: `√(Σ(y_pred - y_actual)²/n)`
   
2. **MAE (Mean Absolute Error)**: Mengukur rata-rata nilai absolut error
   - Formula: `Σ|y_pred - y_actual|/n`
   
3. **Precision**: Proporsi rekomendasi relevan dari semua rekomendasi
   - Formula: `TP/(TP+FP)` (threshold rating = 7.0)
   
4. **Recall**: Proporsi item relevan yang berhasil direkomendasikan
   - Formula: `TP/(TP+FN)`
   
5. **F1-Score**: Harmonic mean dari precision dan recall
   - Formula: `2 × (Precision × Recall)/(Precision + Recall)`
   
6. **User Coverage**: Persentase pengguna yang mendapatkan rekomendasi
   - Formula: `(jumlah_user_dengan_rekomendasi / total_user) × 100%`
   
7. **Item Coverage**: Persentase item yang direkomendasikan
   - Formula: `(jumlah_item_direkomendasikan / total_item) × 100%`
   
8. **Sparsity**: Tingkat kejarangam data dalam matriks user-item
   - Formula: `(1 - jumlah_rating_tersedia / (jumlah_user × jumlah_item)) × 100%`

### Hasil Evaluasi

#### Content-Based Filtering:
- **Diversitas Genre**: 6.10 genre per rekomendasi (dari ~16 genre total)
  - Menunjukkan variasi yang sangat baik dalam rekomendasi
  
- **Cakupan Genre**: 81.25% dari total genre tersedia
  - Sistem tidak bias pada genre tertentu dan mencakup sebagian besar genre
  
- **Rata-rata Skor Kesamaan Konten**: 0.0886
  - Skor ini menunjukkan keseimbangan antara similaritas dan diversitas
  
- **Kualitas Rekomendasi**: 
  - Rating rata-rata: 9.56 (dari skala 10)
  - Standar deviasi: 0.31 (menunjukkan konsistensi kualitas yang tinggi)

#### Collaborative Filtering:
- **RMSE**: 6.9733
  - Nilai error cukup tinggi, menunjukkan adanya deviasi prediksi yang signifikan
  
- **MAE**: 6.6389
  - Rata-rata error absolut yang tinggi
  
- **Precision**: 0.7982 (threshold = 7.0)
  - Hampir 80% dari item yang diprediksi disukai benar-benar disukai pengguna
  
- **Recall**: 0.9988
  - Hampir semua item yang benar-benar disukai pengguna berhasil direkomendasikan
  
- **F1-Score**: 0.8877
  - Keseimbangan yang baik antara precision dan recall
  
- **User Coverage**: 100%
  - Semua pengguna mendapatkan rekomendasi
  
- **Item Coverage**: 100%
  - Semua item berpotensi direkomendasikan
  
- **Sparsity Data**: 85.16%
  - Tingginya sparsity menunjukkan tantangan dalam pemodelan preferensi pengguna

- **Persentase Prediksi dalam Toleransi**:
  - 76.22% prediksi berada dalam jarak 1 poin dari rating aktual
  - 89.68% prediksi berada dalam jarak 2 poin dari rating aktual

### Kesimpulan

1. **Content-Based Filtering** menunjukkan performa yang sangat baik dalam hal diversitas dan kualitas rekomendasi:
   - Diversitas genre yang tinggi (6.10 genre per rekomendasi) menunjukkan sistem mampu merekomendasikan webtoon dari berbagai kategori
   - Cakupan genre yang luas (81.25%) memastikan rekomendasi tidak terbatas pada genre tertentu
   - Rating rata-rata yang tinggi (9.56) dengan standar deviasi rendah (0.31) menunjukkan konsistensi kualitas yang sangat baik
   - Sistem ini sangat cocok untuk merekomendasikan webtoon baru atau untuk pengguna tanpa riwayat preferensi

2. **Collaborative Filtering** menunjukkan kekuatan dalam cakupan namun memiliki tantangan dalam akurasi prediksi:
   - Coverage pengguna dan item sangat baik (100%)
   - Precision (0.7982) dan recall (0.9988) yang tinggi menunjukkan efektivitas dalam mengidentifikasi preferensi pengguna
   - RMSE (6.9733) dan MAE (6.6389) yang tinggi mengindikasikan adanya ketidakakuratan dalam prediksi nilai rating absolut
   - Sparsity data yang tinggi (85.16%) berkontribusi pada tantangan dalam pemodelan preferensi

3. **Rekomendasi Pendekatan**:
   - Untuk pengguna baru atau webtoon baru: Gunakan Content-Based Filtering untuk mengatasi cold-start problem
   - Untuk pengguna dengan riwayat preferensi yang cukup: Gunakan Collaborative Filtering untuk memanfaatkan pola preferensi komunitas
   - Implementasi sistem hybrid yang mengkombinasikan kedua pendekatan dapat menghasilkan rekomendasi yang lebih komprehensif dan akurat

Secara keseluruhan, Content-Based Filtering unggul dalam kualitas rekomendasi dan diversitas, sementara Collaborative Filtering lebih baik dalam menangkap preferensi pengguna, meskipun menghadapi tantangan akurasi prediksi rating absolut.