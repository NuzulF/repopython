from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Pastikan stopwords sudah diunduh
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')

def hitung_skor_cf(df, w1, w2, w3):
    # Konversi kolom ke float
    df["rating"] = df["rating"].astype(float)
    df["sentimen"] = df["sentimen"].astype(float)
    df["jarak"] = df["jarak"].astype(float)

    # normalisasi
    df["rating"] = df["rating"] / 5
    df["sentimen"] = (df['sentimen'] + 1) / 2
    jarak_max = 2000 # estimasi (KM)
    df["jarak"] = 1 - (np.log1p(df['jarak']) / np.log1p(jarak_max))

    # hitung rerata
    df["skor_cf"] = df["rating"] * w1 + df["sentimen"] * w2 + df["jarak"] * w3

    return df

def matriks_cf_user(df):
    # buat pivot table dengan skor_cf sebagai nilai
    df_pivot = df.pivot_table(index='reviewer', columns='nama DTW', values='skor_cf', aggfunc='mean')

    # isi data kosong dengan 0
    df_pivot = df_pivot.fillna(0)

    # hitung cosine similarity / collaborative filtering
    cosine_sim = cosine_similarity(df_pivot)

    # Membuat DataFrame cosine similarity
    cosine_sim_df = pd.DataFrame(cosine_sim, index=df_pivot.index, columns=df_pivot.index)

    return cosine_sim_df, cosine_sim

def lda_ulasan(df, optimal_topics_rev):
    # Preprocessing: Tokenisasi, Stopword Removal
    stop_words = set(stopwords.words('indonesian'))
    df['processed'] = df['review'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.lower() not in stop_words])

    # Buat dictionary dan corpus untuk LDA
    dictionary = corpora.Dictionary(df['processed'])
    corpus = [dictionary.doc2bow(text) for text in df['processed']]

    # Train LDA model
    num_topics = optimal_topics_rev
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Mendapatkan skor topik untuk setiap ulasan
    def get_topic_scores(doc):
        bow = dictionary.doc2bow(doc)
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        return [score for _, score in topic_dist]

    # Mengubah list skor menjadi DataFrame
    lda_scores_matrix = np.array([get_topic_scores(doc) for doc in df['processed']])
    lda_df = pd.DataFrame(lda_scores_matrix, columns=[f'topik{i+1}' for i in range(num_topics)])

    # Menggabungkan dengan reviewer
    final_df = pd.concat([df[['reviewer']], lda_df], axis=1)

    # Agregasi: Hitung rata-rata skor per reviewer
    aggregated_df = final_df.groupby('reviewer').mean().reset_index()

    # isi data kosong dengan 0
    aggregated_df = aggregated_df.fillna(0)

    # Hitung cosine similarity antar pengulas
    cosine_sim_matrix = cosine_similarity(aggregated_df.iloc[:, 1:])  # Hanya kolom topik

    # Konversi ke DataFrame untuk tampilan lebih rapi
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=aggregated_df['reviewer'], columns=aggregated_df['reviewer'])

    return cosine_sim_df, cosine_sim_matrix

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def predict_existing_ratings_user_based(user_id, rating_matrix, similarity_matrix):
    """
    Memprediksi rating untuk wisata yang telah dikunjungi oleh user menggunakan User-Based Collaborative Filtering.

    Parameters:
        user_id (str): ID user yang akan dievaluasi.
        rating_matrix (pd.DataFrame): DataFrame pivot dari skor_cf pengguna-wisata.
        similarity_matrix (pd.DataFrame): DataFrame matriks kesamaan antar pengguna.

    Returns:
        pd.DataFrame: DataFrame berisi rating asli dan prediksi untuk evaluasi.
    """
    # Pastikan user ada dalam rating_matrix
    if user_id not in rating_matrix.index:
        raise ValueError(f"User '{user_id}' tidak ditemukan dalam rating_matrix.")

    # Ambil rating yang telah diberikan oleh user
    user_ratings = rating_matrix.loc[user_id]

    # Pilih hanya wisata yang sudah pernah dikunjungi (rating > 0)
    rated_items = user_ratings.index

    # Ambil daftar user yang mirip dengan user_id (kecuali dirinya sendiri)
    similar_users = similarity_matrix[user_id].drop(user_id, errors="ignore")

    # Pilih top-K user paling mirip
    similar_users = similar_users.sort_values(ascending=False)

    predictions = {}
    for item in rating_matrix.columns:
        # Ambil rating dari user-user yang mirip pada wisata ini
        available_users = rating_matrix[item].dropna().index  # User yang memberikan rating pada wisata ini
        valid_users = similar_users.index.intersection(available_users)

        # Jika tidak ada user mirip yang memberi rating pada wisata ini, lewati prediksi
        if valid_users.empty:
            # print(f"⚠️ Warning: Tidak ada pengguna mirip yang memberi rating pada '{item}'. Melewati item ini.")
            continue

        # Pilih top-K user paling mirip yang telah memberi rating pada wisata ini
        similar_ratings = similar_users.loc[valid_users]

        # Hitung prediksi rating menggunakan weighted sum
        ratings_given = rating_matrix.loc[valid_users, item]
        numerator = np.sum(similar_ratings * ratings_given)
        denominator = np.sum(np.abs(similar_ratings))

        # Pastikan tidak terjadi pembagian dengan nol
        predicted_rating = numerator / denominator if denominator != 0 else np.nan
        predictions[item] = predicted_rating

    # Buat DataFrame hasil prediksi untuk evaluasi
    result_df = pd.DataFrame({
        'Actual': user_ratings[rated_items],
        'Predicted': pd.Series(predictions, index=rated_items)
    })

    return result_df

def evaluate_model(predictions):
    """
    Menghitung RMSE, MAE, dan MAPE berdasarkan prediksi dan rating asli.

    Parameters:
        predictions (pd.DataFrame): DataFrame berisi rating asli dan prediksi.

    Returns:
        dict: RMSE, MAE, dan MAPE dari model.
    """
    predictions = predictions.dropna()  # Hapus prediksi NaN sebelum evaluasi

    actual = predictions['Actual'].values
    predicted = predictions['Predicted'].values

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # Dalam persen

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def predict_for_all_users_user_based(rating_matrix, similarity_matrix, k=5):
    """
    Memanggil fungsi `predict_existing_ratings_user_based` untuk setiap user dan menggabungkan hasilnya.

    Parameters:
        rating_matrix (pd.DataFrame): DataFrame pivot dari skor_cf pengguna-wisata.
        similarity_matrix (pd.DataFrame): DataFrame matriks kesamaan antar pengguna.
        k (int): Jumlah pengguna paling mirip yang akan dipertimbangkan.

    Returns:
        pd.DataFrame: DataFrame berisi rating asli dan prediksi untuk semua user.
    """

    all_predictions = []  # Untuk menyimpan hasil prediksi setiap user

    for user_id in rating_matrix.index:
        print(f"Prediksi untuk User: {user_id}")
        predictions_item = predict_existing_ratings_user_based(user_id, rating_matrix, similarity_matrix, k)
        all_predictions.append(predictions_item)

    # Gabungkan hasil prediksi untuk semua user
    final_predictions = pd.concat(all_predictions)

    return final_predictions