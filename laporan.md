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
1. Bagaimana memberikan rekomendasi webtoon yang relevan berdasarkan konten yang mirip?
2. Bagaimana mengatasi cold-start problem untuk webtoon atau pengguna baru?

### Goals  
1. Mengembangkan sistem rekomendasi content-based yang dapat merekomendasikan webtoon berdasarkan kemiripan genre, penulis, dan ringkasan cerita
2. Mencapai diversitas genre minimal 5 per rekomendasi dengan coverage ≥75%

### Solution Approach
**Content-Based Filtering dengan TF-IDF dan Cosine Similarity**
- Mengatasi cold-start problem
- Memberikan rekomendasi berdasarkan karakteristik konten
- Transparent dan dapat dijelaskan

## Data Understanding

### Sumber Data
Dataset yang digunakan dalam proyek ini adalah Webtoon Comics Dataset yang tersedia di [Kaggle](https://www.kaggle.com/datasets/swarnimrai/webtoon-comics-dataset). Dataset ini berisi informasi tentang webtoon populer dari platform LINE Webtoon.


### Informasi Dataset
- **Jumlah Data**: 569 records  
- **Kondisi Data**: 1 missing value pada kolom Writer
- **Fitur yang Tersedia**: id, Name, Writer, Likes, Genre, Rating, Subscribers, Summary, Update, Reading Link

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

1. **Data Cleaning** - Mengatasi missing values dan memastikan konsistensi data
2. **Feature Engineering** - Menciptakan representasi yang kaya untuk konten webtoon
3. **Text Preprocessing** - Membersihkan dan mempersiapkan data teks untuk analisis
4. **Writer Style Profiling** - Mengidentifikasi pola genre dari penulis untuk memperkaya fitur
5. **TF-IDF Vectorization** - Mengubah fitur teks menjadi representasi vektor numerik
6. **Similarity Matrix Calculation** - Mengukur kesamaan antar item untuk rekomendasi


### Proses Preparation
1. **Data Cleaning**
   - Penanganan Missing Values:
     - Kolom Writer diisi dengan 'Unknown Writer' (1 missing value)
     - Kolom Subscribers_Numeric diisi dengan 0 untuk nilai kosong hasil konversi
   - Standarisasi Format Numerik:
     - Konversi nilai Likes dan Subscribers dari format string ('M', 'K') ke nilai numerik (misal: "5.4M" → 5,400,000)

2. **Text Preprocessing**
   ```python
   df['Summary_Clean'] = df['Summary'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', ' ', x.lower()))
   ```

3. **Feature Engineering**
   ```python
   df['Content_Features'] = df['Genre'] + ' ' + df['Writer'] + ' ' + df['Summary_Clean']
   ```

4. **Writer Style Profiling**
   ```python
   writer_genre_mapping = df.groupby('Writer')['Genre'].agg(lambda x: ' '.join(x)).to_dict()
   df['Writer_Style'] = df['Writer'].map(writer_genre_mapping)
   ```

5. **TF-IDF Vectorization dan Similarity Matrix**
   - **TF-IDF Matrix Shape**: (569, 6118) - 569 webtoon dengan 6118 unique terms
   - **Cosine Similarity Matrix**: (569, 569) untuk mengukur kesamaan antar webtoon

writer_genre_mapping = df.groupby('Writer')['Genre'].agg(lambda x: ' '.join(x)).to_dict()
df['Writer_Style'] = df['Writer'].map(writer_genre_mapping)

## Modeling and Result

### Content-Based Filtering

**Implementasi:**

- Algoritma: TF-IDF Vectorization + Cosine Similarity
- Input: Content_Features (gabungan Genre + Writer + Summary_Clean)
- Output: 10 rekomendasi webtoon dengan similarity score tertinggi

**Cara Kerja:**

1. Mencari indeks webtoon berdasarkan judul
2. Mengambil skor similarity dari cosine_sim matrix
3. Mengurutkan berdasarkan skor tertinggi (tidak termasuk diri sendiri)
4. Mengambil 10 webtoon teratas

### Output Top-N Rekomendasi

#### **Hasil Rekomendasi untuk "Tower of God":**

| Rank | Name | Genre | Rating | Similarity Score |
|------|------|-------|---------|------------------|
| 1. | Tales of Greed | Thriller | 9.33 | 0.164690 |
| 2. | Money Game | Thriller | 8.51 | 0.084385 |
| 3. | Ctrl+Z | Thriller | 7.81 | 0.083716 |
| 4. | She Bites! | Slice of life | 9.70 | 0.062107 |
| 5. | Witch Creek Road | Horror | 9.39 | 0.055339 |
| 6. | Warrior of the Wild | Action | 9.46 | 0.052517 |
| 7. | Tales of Sarimin the Jinn | Action | 9.54 | 0.048883 |
| 8. | Cape of Spirits | Action | 9.50 | 0.048561 |
| 9. | Athena Complex | Fantasy | 9.54 | 0.048340 |
| 10. | My Dear Cold-Blooded King | Romance | 9.72 | 0.046575 |

**Analisis**: 
- Rating rata-rata: 9.16
- Genre unik: 6 dari 10 rekomendasi
- Genre yang muncul: Thriller, Slice of life, Horror, Action, Fantasy, Romance
- Similarity score: 0.046575 - 0.164690

#### **Hasil Rekomendasi untuk "Omniscient Reader":**

| Rank | Name | Genre | Rating | Similarity Score |
|------|------|-------|---------|------------------|
| 1. | The First Night With the Duke | Fantasy | 9.63 | 0.155865 |
| 2. | Happily Ever Afterwards | Fantasy | 9.65 | 0.143122 |
| 3. | Half-Ghost | Romance | 9.74 | 0.132341 |
| 4. | Elena | Horror | 9.52 | 0.125274 |
| 5. | Good Luck, Hero! | Fantasy | 9.58 | 0.122048 |
| 6. | Surviving Romance | Horror | 9.81 | 0.118337 |
| 7. | How to Survive a Romance Fantasy | Fantasy | 9.41 | 0.101325 |
| 8. | The Male Lead's Girl Friend | Romance | 9.33 | 0.100728 |
| 9. | Tricked into Becoming the Heroine's Stepmother | Fantasy | 9.61 | 0.096719 |
| 10. | Back to You | Drama | 9.70 | 0.094367 |

**Analisis**:
- Rating rata-rata: 9.60
- Genre unik: 5 dari 10 rekomendasi
- Genre yang muncul: Fantasy, Romance, Horror, Drama
- Similarity score: 0.094367 - 0.155865

#### **Hasil Rekomendasi untuk "Eleceed":**

| Rank | Name | Genre | Rating | Similarity Score |
|------|------|-------|---------|------------------|
| 1. | Fluffy Boyfriend | Supernatural | 9.13 | 0.142221 |
| 2. | Saphie: The One-Eyed Cat | Slice of life | 9.80 | 0.111078 |
| 3. | Watermelon | Fantasy | 9.80 | 0.097568 |
| 4. | Meow Man | Comedy | 9.70 | 0.097199 |
| 5. | Meow Rangers | Comedy | 9.34 | 0.092227 |
| 6. | I Get Stronger the More I Eat | Action | 8.45 | 0.082215 |
| 7. | UnderPrin | Supernatural | 9.28 | 0.079013 |
| 8. | My Deepest Secret | Thriller | 9.68 | 0.077729 |
| 9. | WEBTOON Now | Informative | 9.25 | 0.076889 |
| 10. | Noblesse | Action | 9.77 | 0.074212 |

**Analisis**:
- Rating rata-rata: 9.42
- Genre unik: 8 dari 10 rekomendasi
- Genre yang muncul: Supernatural, Slice of life, Fantasy, Comedy, Action, Thriller, Informative
- Similarity score: 0.074212 - 0.142221

### Insight dari Output Rekomendasi:

1. **Konsistensi Kualitas**: Rating rata-rata semua rekomendasi > 9.0 menunjukkan sistem berhasil merekomendasikan webtoon berkualitas tinggi
2. **Diversitas Genre**: Setiap rekomendasi menampilkan 5-8 genre unik, menunjukkan kemampuan sistem dalam memberikan variasi
3. **Similarity Score**: Rentang 0.07-0.17 menunjukkan keseimbangan antara relevansi dan diversitas
4. **Adaptabilitas**: Sistem dapat merekomendasikan berbagai jenis webtoon berdasarkan karakteristik input yang berbeda
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

## Evaluation

### Metrik Evaluasi

#### Content-Based Filtering:
1. **Diversitas Genre**: Rata-rata jumlah genre unik dalam rekomendasi
2. **Coverage Genre**: Persentase genre yang tercakup
3. **Kualitas Rekomendasi**: Rating rata-rata dan konsistensi
4. **Rata-rata Skor Kesamaan Konten**: Tingkat kesamaan dalam rekomendasi

### Hasil Evaluasi

#### Content-Based Filtering:
- **Diversitas Genre**: 6.10 genre per rekomendasi (dari ~16 genre total)
  - Menunjukkan variasi yang sangat baik dalam rekomendasi
  
- **Cakupan Genre**: 81.25% dari total genre tersedia
  - Sistem tidak bias pada genre tertentu dan mencakup sebagian besar genre
  
- **Rata-rata Skor Kesamaan Konten**: 0.0840
  - Skor ini menunjukkan keseimbangan antara similaritas dan diversitas
  
- **Kualitas Rekomendasi**: 
  - Rating rata-rata: 9.56 (dari skala 10)
  - Standar deviasi: 0.31 (menunjukkan konsistensi kualitas yang tinggi)

### Kesimpulan

**Content-Based Filtering** menunjukkan performa yang sangat baik dalam hal diversitas dan kualitas rekomendasi:
   - Diversitas genre yang tinggi (6.10 genre per rekomendasi) menunjukkan sistem mampu merekomendasikan webtoon dari berbagai kategori
   - Cakupan genre yang luas (81.25%) memastikan rekomendasi tidak terbatas pada genre tertentu
   - Rating rata-rata yang tinggi (9.56) dengan standar deviasi rendah (0.31) menunjukkan konsistensi kualitas yang sangat baik
   - Sistem ini sangat cocok untuk merekomendasikan webtoon baru atau untuk pengguna tanpa riwayat preferensi

Secara keseluruhan, Content-Based Filtering unggul dalam kualitas rekomendasi dan diversitas.