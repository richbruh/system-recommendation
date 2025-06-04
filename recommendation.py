# Cell 1: Project Overview
"""
# Sistem Rekomendasi Webtoon - Content-Based Filtering
    
## Latar Belakang
Webtoon telah menjadi salah satu bentuk hiburan digital yang sangat populer. 
Dengan ribuan judul yang tersedia di platform seperti LINE Webtoon, pengguna 
sering kesulitan menemukan webtoon yang sesuai dengan preferensi mereka. 
Sistem rekomendasi content-based dapat membantu pengguna menemukan webtoon 
baru berdasarkan kesamaan konten seperti genre, penulis, dan ringkasan cerita, 
tanpa memerlukan data interaksi pengguna sebelumnya.

### Mengapa Proyek Ini Penting?
1. Meningkatkan User Experience: Memudahkan pengguna menemukan konten yang sesuai dengan preferensi mereka
2. Mengatasi Information Overload: Membantu pengguna menavigasi ribuan judul webtoon yang tersedia  
3. Diversifikasi Konten: Memperkenalkan pengguna pada genre yang mungkin belum mereka kenal
4. Membantu Kreator: Meningkatkan peluang karya berkualitas namun kurang populer untuk ditemukan

### Problem Statement
Bagaimana memberikan rekomendasi webtoon yang relevan berdasarkan konten yang mirip dan mengatasi cold-start problem untuk webtoon atau pengguna baru?

### Goals
1. Mengembangkan sistem rekomendasi content-based yang dapat merekomendasikan webtoon berdasarkan kemiripan genre, penulis, dan ringkasan cerita
2. Mencapai diversitas genre minimal 5 per rekomendasi dengan coverage ≥75%
3. Memberikan rekomendasi berkualitas tinggi dengan rating rata-rata ≥9.0

### Solution Approach
Content-Based Filtering dengan TF-IDF dan Cosine Similarity
- Mengatasi cold-start problem untuk item baru
- Memberikan rekomendasi berdasarkan karakteristik konten
- Transparent dan dapat dijelaskan (explainable AI)
- Konsistensi dalam memberikan rekomendasi
"""

# Cell 2: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Cell 3: Data Understanding
"""
#### 2. Data Understanding

Dataset yang digunakan dalam proyek ini adalah Webtoon Comics Dataset yang tersedia di [Kaggle](https://www.kaggle.com/datasets/swarnimrai/webtoon-comics-dataset). Dataset ini berisi informasi tentang berbagai webtoon populer dari platform LINE Webtoon.

## 2.1 Data Loading

#### Informasi Dataset
- **Jumlah Data**: 569 records
- **Kondisi Data**: 1 missing value pada kolom Writer 
- **Fitur yang Tersedia**: id, Name, Writer, Likes, Genre, Rating, Subscribers, Summary, Update, Reading Link

#### Variabel/Fitur
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
"""

# Cell 4: Data Loading
df = pd.read_csv('webtoon-dataset.csv')

# Display basic information
print("Jumlah data:", len(df))
print("\nInformasi Dataset:")
print(df.info())
print("\nJumlah Baris:", df.shape[0])
print("Jumlah Kolom:", df.shape[1])

print("\nSample data:")
print(df.head())

print("\nDeskripsi statistik:")
print(df.describe())

# Cell 5: Exploratory Data Analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# 3.1 Missing Values dan Duplikat Data
print("\nJumlah Missing Values:")
print(df.isnull().sum())
print("\nJumlah Duplikat:")
print(df.duplicated().sum())

# 3.2 Distribusi Genre
plt.figure(figsize=(10, 5))
genre_counts = df['Genre'].value_counts()
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.title('Distribusi Genre Webtoon')
plt.xlabel('Genre')
plt.ylabel('Jumlah')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(f"Jumlah unik genre: {len(genre_counts)}")
print("Genre terpopuler: ", genre_counts.index[0], "dengan", genre_counts.values[0], "webtoon")

# 3.3 Distribusi Rating
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=10, kde=True)
plt.title('Distribusi Rating Webtoon')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.axvline(df['Rating'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {df["Rating"].mean():.2f}')
plt.axvline(df['Rating'].median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {df["Rating"].median():.2f}')
plt.legend()
plt.show()

print(f"Rating tertinggi: {df['Rating'].max():.2f}")
print(f"Rating terendah: {df['Rating'].min():.2f}")
print(f"Rating rata-rata: {df['Rating'].mean():.2f}")

# 3.4 Hubungan Genre dengan Rating
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Rating', data=df)
plt.title('Distribusi Rating per Genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3.5 Popularity Analysis
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.strip()
        if 'M' in value:
            return float(value.replace('M', '')) * 1000000
        elif 'K' in value:
            return float(value.replace('K', '')) * 1000
        else:
            try:
                return float(value.replace(',', ''))
            except:
                return None
    return value

# Create clean numeric columns
df['Likes_Numeric'] = df['Likes'].apply(convert_to_numeric)
df['Subscribers_Numeric'] = df['Subscribers'].apply(convert_to_numeric)

# Top Webtoons by Likes
top_likes = df.sort_values(by='Likes_Numeric', ascending=False).head(10)
print("\nTop 10 Most-Liked Webtoons:")
print(top_likes[['Name', 'Genre', 'Likes', 'Likes_Numeric', 'Rating']].reset_index(drop=True))

# Top Webtoons by Subscribers
top_subscribers = df.sort_values(by='Subscribers_Numeric', ascending=False).head(10)
print("\nTop 10 Webtoons by Subscriber Count:")
print(top_subscribers[['Name', 'Genre', 'Subscribers', 'Subscribers_Numeric', 'Rating']].reset_index(drop=True))

# Top Webtoons by Rating (with minimum 100K subscribers for significance)
top_rated = df[df['Subscribers_Numeric'] >= 100000].sort_values(by='Rating', ascending=False).head(10)
print("\nTop 10 Highest-Rated Popular Webtoons (min 100K subscribers):")
print(top_rated[['Name', 'Genre', 'Rating', 'Subscribers', 'Likes']].reset_index(drop=True))

# Genre Popularity Analysis
genre_popularity = df.groupby('Genre').agg({
    'Likes_Numeric': 'mean',
    'Subscribers_Numeric': 'mean',
    'Rating': 'mean',
    'id': 'count'
}).sort_values(by='Subscribers_Numeric', ascending=False)

genre_popularity.columns = ['Avg Likes', 'Avg Subscribers', 'Avg Rating', 'Webtoon Count']
print("\nGenre Popularity (Sorted by Average Subscribers):")
print(genre_popularity.head(10))

# Correlation analysis
print("\nCorrelation between Popularity Metrics:")
correlation = df[['Likes_Numeric', 'Subscribers_Numeric', 'Rating']].corr()
print(correlation)

# Update Schedule vs. Popularity
update_popularity = df.groupby('Update').agg({
    'Likes_Numeric': 'mean',
    'Subscribers_Numeric': 'mean',
    'Rating': 'mean',
    'id': 'count'
}).sort_values(by='Subscribers_Numeric', ascending=False)

update_popularity.columns = ['Avg Likes', 'Avg Subscribers', 'Avg Rating', 'Webtoon Count']
print("\nUpdate Schedule vs. Popularity (Sorted by Average Subscribers):")
print(update_popularity.head(10))

# Status Analysis (Completed vs. Ongoing)
df['Status'] = df['Update'].apply(lambda x: 'Completed' if x == 'COMPLETED' else 'Ongoing')
status_popularity = df.groupby('Status').agg({
    'Likes_Numeric': 'mean',
    'Subscribers_Numeric': 'mean',
    'Rating': 'mean',
    'id': 'count'
})

status_popularity.columns = ['Avg Likes', 'Avg Subscribers', 'Avg Rating', 'Webtoon Count']
print("\nCompleted vs. Ongoing Webtoons Popularity:")
print(status_popularity)

# Writer Popularity (top writers with multiple webtoons)
writer_counts = df['Writer'].value_counts()
top_writers = writer_counts[writer_counts > 1].index.tolist()
if top_writers:
    writer_popularity = df[df['Writer'].isin(top_writers)].groupby('Writer').agg({
        'Likes_Numeric': 'mean',
        'Subscribers_Numeric': 'mean',
        'Rating': 'mean',
        'id': 'count'
    }).sort_values(by='Subscribers_Numeric', ascending=False)
    
    writer_popularity.columns = ['Avg Likes', 'Avg Subscribers', 'Avg Rating', 'Webtoon Count']
    print("\nTop Writers with Multiple Webtoons (by Average Subscribers):")
    print(writer_popularity.head(10))

# Visualizations
# Top 10 Webtoons by Subscribers (Bar Chart)
plt.figure(figsize=(12, 6))
sns.barplot(x='Name', y='Subscribers_Numeric', data=top_subscribers)
plt.title('Top 10 Webtoons by Subscriber Count')
plt.xlabel('Webtoon Title')
plt.ylabel('Subscribers (millions)')
plt.xticks(rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()

# Genre Popularity (by average subscribers)
plt.figure(figsize=(12, 6))
genre_sub_plot = genre_popularity.head(10).reset_index()
sns.barplot(x='Genre', y='Avg Subscribers', data=genre_sub_plot)
plt.title('Average Subscribers by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Subscribers')
plt.xticks(rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Popularity Metrics')
plt.tight_layout()
plt.show()

# Scatter plot: Subscribers vs. Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Subscribers_Numeric', hue='Genre', data=df, alpha=0.7)
plt.title('Relationship Between Rating and Subscribers')
plt.xlabel('Rating')
plt.ylabel('Subscribers')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Completed vs. Ongoing Comparison
plt.figure(figsize=(10, 5))
status_df = status_popularity.reset_index()
status_df = pd.melt(status_df, id_vars=['Status'], 
                    value_vars=['Avg Likes', 'Avg Subscribers', 'Avg Rating'],
                    var_name='Metric', value_name='Value')
sns.barplot(x='Status', y='Value', hue='Metric', data=status_df)
plt.title('Completed vs. Ongoing Webtoons Comparison')
plt.xlabel('Status')
plt.ylabel('Average Value (Normalized)')
plt.tight_layout()
plt.show()

# Print summary insights
print("\n====== POPULARITY INSIGHTS ======")
print(f"Most popular webtoon by subscribers: {top_subscribers['Name'].iloc[0]} ({top_subscribers['Subscribers_Numeric'].iloc[0]/1000000:.2f}M)")
print(f"Most popular genre by avg subscribers: {genre_popularity.index[0]}")
print(f"Most effective update schedule: {update_popularity.index[0]}")
print(f"Correlation between Likes and Subscribers: {correlation.loc['Likes_Numeric', 'Subscribers_Numeric']:.2f}")
print(f"Correlation between Rating and Subscribers: {correlation.loc['Rating', 'Subscribers_Numeric']:.2f}")

# Calculate the percentage of top 10 subscribers that belong to each genre
top10_genre_counts = top_subscribers['Genre'].value_counts()
print("\nGenre distribution in Top 10 most subscribed:")
for genre, count in top10_genre_counts.items():
    print(f"- {genre}: {count} webtoons ({count*10}%)")

print(f"\nRating Tertinggi: {df['Rating'].max():.2f}")
print(f"Rating Terendah: {df['Rating'].min():.2f}")
print(f"Rating Rata-rata: {df['Rating'].mean():.2f}")

# Cell 6: Data Preparation
print("\n" + "="*50)
print("DATA PREPARATION")
print("="*50)

"""
#### Teknik yang Digunakan
1. **Data Cleaning** - Mengatasi missing values dan memastikan konsistensi data
2. **Feature Engineering** - Menciptakan representasi yang kaya untuk konten webtoon
3. **Text Preprocessing** - Membersihkan dan mempersiapkan data teks untuk analisis
4. **Writer Style Profiling** - Mengidentifikasi pola genre dari penulis untuk memperkaya fitur
5. **TF-IDF Vectorization** - Mengubah fitur teks menjadi representasi vektor numerik
6. **Similarity Matrix Calculation** - Mengukur kesamaan antar item untuk rekomendasi
"""

# 4.1 Data Cleaning
print("Missing values before Cleaning:")
print(df.isnull().sum())

# Print baris yang memiliki missing values
missing_rows = df[df.isnull().any(axis=1)]
print("\nRows with Missing Values:")
print(missing_rows)

# Isi Missing Values dengan 'Unknown'
df['Writer'] = df['Writer'].fillna('Unknown Writer')
df['Genre'] = df['Genre'].fillna('Uncategorized')
df['Subscribers_Numeric'] = df['Subscribers_Numeric'].fillna(0)

print("\nMissing values after Cleaning:")
print(df.isnull().sum())

print("\nSample data after cleaning missing values:")
print(df[['Name', 'Writer', 'Genre', 'Likes_Numeric', 'Subscribers_Numeric']].head())

# 4.2 Feature Engineering
print("\n4.2.1 Text Preprocessing untuk Ringkasan")
# Clean and process summary text
df['Summary_Clean'] = df['Summary'].fillna('').apply(lambda x: re.sub(r'[^\w\s]', ' ', x.lower()))

print("\n4.2.2 Penggabungan Fitur untuk Representasi Konten")
# Create a single rich feature combining multiple attributes
df['Content_Features'] = df['Genre'] + ' ' + df['Writer'] + ' ' + df['Summary_Clean']

print("\n4.2.3 Profil Gaya Penulis")
# Create writer style profiles based on their existing works
writer_genre_mapping = df.groupby('Writer')['Genre'].agg(lambda x: ' '.join(x)).to_dict()
df['Writer_Style'] = df['Writer'].map(writer_genre_mapping)

# 4.3 Vectorization dan Similarity Matrix
print("\n4.3 Vectorization dan Similarity Matrix untuk Content Based Filtering")
print("Transformasi fitur teks menjadi vektor TF-IDF...")

# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Transformasi Content_Features menjadi matriks TF-IDF
tfidf_matrix = tfidf.fit_transform(df['Content_Features'])

print(f"Bentuk matriks TF-IDF: {tfidf_matrix.shape}")
print(f"Jumlah feature/terms dalam kosakata: {len(tfidf.get_feature_names_out())}")

# Menampilkan beberapa feature/terms teratas
print("\nBeberapa feature/terms dari TF-IDF vocabulary:")
print(tfidf.get_feature_names_out()[:10])

# Menghitung cosine similarity matrix
print("\nMenghitung cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Bentuk similarity matrix: {cosine_sim.shape}")

# Membuat dictionary untuk mapping id webtoon ke index
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

# Fungsi untuk mendapatkan rekomendasi berdasarkan judul webtoon
def get_recommendations(title, cosine_sim=cosine_sim):
    """
    Berikan rekomendasi webtoon berdasarkan kesamaan konten dengan judul yang diberikan
    
    Parameters:
    title (str): Judul webtoon yang menjadi acuan rekomendasi
    cosine_sim (numpy.ndarray): Matrix cosine similarity
    
    Returns:
    pandas.DataFrame: DataFrame berisi 10 rekomendasi webtoon teratas
    """
    try:
        # Dapatkan index webtoon yang sesuai dengan judul
        idx = indices[title]
        
        # Dapatkan skor kesamaan untuk semua webtoon dengan webtoon tersebut
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Urutkan webtoon berdasarkan skor kesamaan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil 11 webtoon dengan skor tertinggi (yang pertama adalah webtoon itu sendiri)
        sim_scores = sim_scores[1:11]
        
        # Dapatkan indeks webtoon
        webtoon_indices = [i[0] for i in sim_scores]
        
        # Kembalikan 10 webtoon teratas dengan skor similaritynya
        result = df.iloc[webtoon_indices][['Name', 'Genre', 'Writer', 'Rating']].copy()
        result['Similarity Score'] = [i[1] for i in sim_scores]
        return result
    
    except KeyError:
        print(f"Judul '{title}' tidak ditemukan dalam dataset.")
        return None

# Contoh rekomendasi untuk sebuah webtoon populer
sample_webtoon = df['Name'].iloc[0]  # Ambil judul webtoon pertama sebagai contoh
print(f"\nContoh rekomendasi untuk webtoon '{sample_webtoon}':")
recommendations = get_recommendations(sample_webtoon)
if recommendations is not None:
    print(recommendations)

# Visualisasi Similarity Matrix
plt.figure(figsize=(10, 8))
plt.title("Heatmap Similarity Matrix (10 Webtoon Pertama)")
sns.heatmap(cosine_sim[:10, :10], annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=df['Name'][:10], yticklabels=df['Name'][:10])
plt.tight_layout()
plt.show()

# Simpan hasil vectorization dan similarity matrix untuk digunakan dalam model
content_based_data = {
    'tfidf_vectorizer': tfidf,
    'tfidf_matrix': tfidf_matrix,
    'cosine_sim': cosine_sim,
    'indices': indices
}

print("\nVectorization dan similarity matrix berhasil dibuat dan disimpan untuk modeling.")

# Cell 7: Content-Based Filtering
print("\n" + "="*50)
print("MODELING")
print("="*50)

print("===== Content-Based Filtering Model =====\n")

# Test beberapa webtoon untuk rekomendasi
test_webtoons = ['Tower of God', 'Omniscient Reader', 'Eleceed']

for webtoon in test_webtoons:
    if webtoon in df['Name'].values:
        print(f"\nRecommendations for '{webtoon}':")
        recommendations = get_recommendations(webtoon)
        print(recommendations)
    else:
        print(f"\nWebtoon '{webtoon}' not found in dataset.")

# Visualize recommendations for one example
def plot_recommendations(title):
    if title in df['Name'].values:
        # Get recommendations
        recommendations = get_recommendations(title)
        
        # Get original webtoon info
        original = df[df['Name'] == title].iloc[0]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(recommendations['Name'], recommendations['Similarity Score'], color='skyblue')
        plt.xlabel('Similarity Score')
        plt.ylabel('Recommended Webtoons')
        plt.title(f'Top 10 Recommendations for "{title}" (Genre: {original["Genre"]})')
        plt.tight_layout()
        plt.show()
        
        # Print genre comparison
        print("\nGenre Comparison:")
        original_genre = original['Genre']
        for _, rec in recommendations.iterrows():
            print(f"{rec['Name']} - Genre: {rec['Genre']} (Original: {original_genre})")
            
sample_webtoon = df['Name'].iloc[1]  # Use second webtoon as example
plot_recommendations(sample_webtoon)

# Cell 8: Evaluation
print("\n" + "="*50)
print("EVALUATION")
print("="*50)

"""
## 6. Evaluation

### Metrik Evaluasi Content-Based Filtering

#### 1. Diversitas Genre  
Mengukur rata-rata jumlah genre unik dalam rekomendasi.
- **Formula**: `Diversitas = (1/N) × Σᵢ|Gᵢ|`

#### 2. Coverage Genre
Mengukur persentase genre yang tercakup dalam rekomendasi.
- **Formula**: `Coverage = (|G_rec| / |G_total|) × 100%`

#### 3. Kualitas Rekomendasi
Mengukur rating rata-rata dan konsistensi webtoon yang direkomendasikan.
- **Formula**: `Avg(ratings)`, `Std(ratings)`

#### 4. Rata-rata Skor Kesamaan Konten
Mengukur tingkat kesamaan rata-rata dalam rekomendasi.
- **Formula**: `Avg(similarity_scores)`
"""

print("\n===== EVALUASI CONTENT-BASED FILTERING =====\n")

# Metrik evaluasi untuk Content-Based Filtering
def evaluate_content_based_filtering():
    """
    Evaluasi model Content-Based Filtering menggunakan berbagai metrik
    """
    print("Mengevaluasi performa Content-Based Filtering...")
    
    # 1. Evaluasi Diversitas Rekomendasi
    diversity_scores = []
    coverage_genres = set()
    
    # Ambil sampel webtoon untuk evaluasi
    sample_webtoons = df['Name'].head(10).tolist()
    
    for webtoon in sample_webtoons:
        if webtoon in df['Name'].values:
            recommendations = get_recommendations(webtoon)
            if recommendations is not None:
                # Hitung diversitas genre dalam rekomendasi
                rec_genres = recommendations['Genre'].unique()
                diversity_scores.append(len(rec_genres))
                coverage_genres.update(rec_genres)
    
    avg_diversity = np.mean(diversity_scores)
    genre_coverage = len(coverage_genres) / len(df['Genre'].unique()) * 100
    
    print(f"Rata-rata diversitas genre per rekomendasi: {avg_diversity:.2f}")
    print(f"Cakupan genre dalam rekomendasi: {genre_coverage:.2f}%")
    
    # 2. Evaluasi Kesamaan Konten
    similarity_scores = []
    
    for webtoon in sample_webtoons[:5]:  # Ambil 5 sampel
        if webtoon in df['Name'].values:
            recommendations = get_recommendations(webtoon)
            if recommendations is not None:
                avg_similarity = recommendations['Similarity Score'].mean()
                similarity_scores.append(avg_similarity)
    
    avg_content_similarity = np.mean(similarity_scores)
    print(f"Rata-rata skor kesamaan konten: {avg_content_similarity:.4f}")
    
    # 3. Evaluasi Distribusi Rating Rekomendasi
    recommended_ratings = []
    
    for webtoon in sample_webtoons:
        if webtoon in df['Name'].values:
            recommendations = get_recommendations(webtoon)
            if recommendations is not None:
                recommended_ratings.extend(recommendations['Rating'].tolist())
    
    avg_recommended_rating = np.mean(recommended_ratings)
    std_recommended_rating = np.std(recommended_ratings)
    
    print(f"Rata-rata rating webtoon yang direkomendasikan: {avg_recommended_rating:.2f}")
    print(f"Standar deviasi rating rekomendasi: {std_recommended_rating:.2f}")
    
    return {
        'avg_diversity': avg_diversity,
        'genre_coverage': genre_coverage,
        'avg_similarity': avg_content_similarity,
        'avg_recommended_rating': avg_recommended_rating,
        'std_recommended_rating': std_recommended_rating
    }

# Jalankan evaluasi Content-Based Filtering
cb_metrics = evaluate_content_based_filtering()

# Visualisasi hasil evaluasi Content-Based
def visualize_content_based_evaluation():
    """
    Visualisasi hasil evaluasi Content-Based Filtering
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribusi Similarity Score
    sample_webtoons = df['Name'].head(5).tolist()
    all_similarities = []
    
    for webtoon in sample_webtoons:
        if webtoon in df['Name'].values:
            recommendations = get_recommendations(webtoon)
            if recommendations is not None:
                all_similarities.extend(recommendations['Similarity Score'].tolist())
    
    axes[0, 0].hist(all_similarities, bins=15, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribusi Skor Kesamaan Content-Based')
    axes[0, 0].set_xlabel('Skor Kesamaan')
    axes[0, 0].set_ylabel('Frekuensi')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribusi Genre dalam Rekomendasi
    all_genres = []
    for webtoon in sample_webtoons:
        if webtoon in df['Name'].values:
            recommendations = get_recommendations(webtoon)
            if recommendations is not None:
                all_genres.extend(recommendations['Genre'].tolist())
    
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    axes[0, 1].bar(range(len(genre_counts)), genre_counts.values, color='lightcoral')
    axes[0, 1].set_title('10 Genre Teratas dalam Rekomendasi')
    axes[0, 1].set_xlabel('Genre')
    axes[0, 1].set_ylabel('Frekuensi')
    axes[0, 1].set_xticks(range(len(genre_counts)))
    axes[0, 1].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    
    # 3. Rating vs Similarity Score
    ratings = []
    similarities = []
    
    for webtoon in sample_webtoons:
        if webtoon in df['Name'].values:
            recommendations = get_recommendations(webtoon)
            if recommendations is not None:
                ratings.extend(recommendations['Rating'].tolist())
                similarities.extend(recommendations['Similarity Score'].tolist())
    
    axes[1, 0].scatter(similarities, ratings, alpha=0.6, color='green')
    axes[1, 0].set_title('Korelasi Skor Kesamaan vs Rating')
    axes[1, 0].set_xlabel('Skor Kesamaan')
    axes[1, 0].set_ylabel('Rating')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrik Evaluasi
    metrics_names = ['Diversitas\nGenre', 'Cakupan\nGenre (%)', 'Rata-rata\nSimilarity', 'Rata-rata\nRating']
    metrics_values = [cb_metrics['avg_diversity'], cb_metrics['genre_coverage'], 
                     cb_metrics['avg_similarity'] * 10, cb_metrics['avg_recommended_rating']]  # Scale similarity for visibility
    
    bars = axes[1, 1].bar(metrics_names, metrics_values, color=['purple', 'orange', 'brown', 'pink'])
    axes[1, 1].set_title('Ringkasan Metrik Content-Based Filtering')
    axes[1, 1].set_ylabel('Nilai')
    
    # Tambahkan nilai di atas bar
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

visualize_content_based_evaluation()

# Cell 9: Kesimpulan
print("\n" + "="*50)
print("KESIMPULAN")
print("="*50)

"""
## 7. Kesimpulan

### Ringkasan Pencapaian

Content-Based Filtering menunjukkan performa yang sangat baik:

**Hasil Evaluasi:**
- ✅ **Diversitas Genre**: 6.10 genre per rekomendasi menunjukkan variasi yang sangat baik
- ✅ **Cakupan Genre**: 81.25% coverage menunjukkan sistem tidak bias pada genre tertentu  
- ✅ **Kualitas Rekomendasi**: Rating rata-rata 9.56 dengan standar deviasi rendah (0.31)
- ✅ **Rata-rata Skor Kesamaan**: 0.0840 menunjukkan keseimbangan antara similaritas dan diversitas

**Kelebihan Content-Based Filtering:**
- No Cold-Start Problem untuk item baru
- Transparansi dan explainability tinggi
- User Independence
- Konsistensi temporal

**Kekurangan Content-Based Filtering:**
- Limited content analysis
- Potensi over-specialization 
- Dependency pada feature engineering
- Kurang serendipity

**Rekomendasi Implementasi:**
Sistem ini ideal untuk platform webtoon yang ingin memberikan rekomendasi 
berdasarkan konten, terutama untuk pengguna baru atau webtoon yang baru diluncurkan.
"""

print("Content-Based Filtering menunjukkan performa yang sangat baik:")
print(f"- Diversitas Genre: {cb_metrics['avg_diversity']:.2f} genre per rekomendasi")
print(f"- Cakupan Genre: {cb_metrics['genre_coverage']:.2f}% coverage")
print(f"- Kualitas Rekomendasi: Rating rata-rata {cb_metrics['avg_recommended_rating']:.2f}")
print(f"- Rata-rata Skor Kesamaan: {cb_metrics['avg_similarity']:.4f}")

print("\nSistem rekomendasi Content-Based Filtering telah berhasil diimplementasikan!")
print("Sistem ini ideal untuk platform webtoon yang ingin memberikan rekomendasi")
print("berdasarkan konten, terutama untuk pengguna baru atau webtoon yang baru diluncurkan.")