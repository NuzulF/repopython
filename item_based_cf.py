from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

def lda_ulasan_item(df, optimal_topics_rev):
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

    # Menggabungkan dengan nama DTW
    final_df = pd.concat([df[['nama DTW']], lda_df], axis=1)

    # Agregasi: Hitung rata-rata skor per nama DTW
    aggregated_df = final_df.groupby('nama DTW').mean().reset_index()

    # isi data kosong dengan 0
    aggregated_df = aggregated_df.fillna(0)

    # # Hitung cosine similarity antar pengulas
    # cosine_sim_matrix = cosine_similarity(aggregated_df.iloc[:, 1:])  # Hanya kolom topik

    # # Konversi ke DataFrame untuk tampilan lebih rapi
    # cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=aggregated_df['nama DTW'], columns=aggregated_df['nama DTW'])

    return aggregated_df

def lda_ulasan_item_des(df, optimal_topics_des):
    # Preprocessing: Tokenisasi, Stopword Removal
    stop_words = set(stopwords.words('indonesian'))
    df['processed'] = df['deskripsi'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.lower() not in stop_words])

    # Buat dictionary dan corpus untuk LDA
    dictionary = corpora.Dictionary(df['processed'])
    corpus = [dictionary.doc2bow(text) for text in df['processed']]

    # Train LDA model
    num_topics = optimal_topics_des
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Mendapatkan skor topik untuk setiap ulasan
    def get_topic_scores(doc):
        bow = dictionary.doc2bow(doc)
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
        return [score for _, score in topic_dist]

    # Mengubah list skor menjadi DataFrame
    lda_scores_matrix = np.array([get_topic_scores(doc) for doc in df['processed']])
    lda_df = pd.DataFrame(lda_scores_matrix, columns=[f'topik_des{i+1}' for i in range(num_topics)])

    # Menggabungkan dengan nama DTW
    final_df = pd.concat([df[['nama DTW']], lda_df], axis=1)

    # Agregasi: Hitung rata-rata skor per nama DTW
    aggregated_df = final_df.groupby('nama DTW').mean().reset_index()

    # isi data kosong dengan 0
    aggregated_df = aggregated_df.fillna(0)

    # # Hitung cosine similarity antar pengulas
    # cosine_sim_matrix = cosine_similarity(aggregated_df.iloc[:, 1:])  # Hanya kolom topik

    # # Konversi ke DataFrame untuk tampilan lebih rapi
    # cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=aggregated_df['nama DTW'], columns=aggregated_df['nama DTW'])

    return aggregated_df

def item_based_similarity(df):
    # Hitung cosine similarity antar pengulas
    cosine_sim_matrix = cosine_similarity(df.iloc[:, 1:])  # Hanya kolom topik

    # Konversi ke DataFrame untuk tampilan lebih rapi
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df['nama DTW'], columns=df['nama DTW'])

    return cosine_sim_df, cosine_sim_matrix

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict_existing_ratings(user_id, rating_matrix, similarity_matrix, k=5):
    """
    Memprediksi rating untuk wisata yang telah dikunjungi oleh user menggunakan Item-Based Collaborative Filtering.

    Parameters:
        user_id (str): ID user yang akan dievaluasi.
        rating_matrix (pd.DataFrame): DataFrame berisi reviewer, nama DTW, dan skor_cf.
        similarity_matrix (pd.DataFrame): DataFrame matriks kesamaan antar wisata.
        k (int): Jumlah wisata paling mirip yang akan dipertimbangkan.

    Returns:
        pd.DataFrame: DataFrame berisi rating asli dan prediksi untuk evaluasi.
    """

    # Normalisasi nama wisata agar seragam
    rating_matrix['nama DTW'] = rating_matrix['nama DTW'].str.strip().str.lower()
    similarity_matrix.index = similarity_matrix.index.str.strip().str.lower()
    similarity_matrix.columns = similarity_matrix.columns.str.strip().str.lower()

    # Buat pivot table dengan skor_cf sebagai nilai
    rating_pivot = rating_matrix.pivot_table(index='reviewer', columns='nama DTW', values='skor_cf', aggfunc='mean')

    # Isi data kosong dengan 0
    rating_pivot = rating_pivot.fillna(0)

    # Pastikan user ada dalam rating_pivot
    if user_id not in rating_pivot.index:
        raise ValueError(f"User '{user_id}' tidak ditemukan dalam rating_matrix.")

    # Ambil rating yang telah diberikan oleh user
    user_ratings = rating_pivot.loc[user_id]

    # Pilih hanya wisata yang sudah pernah dikunjungi (rating > 0)
    rated_items = user_ratings.index

    predictions = {}
    for item in rating_matrix['nama DTW'].values:
        if item not in similarity_matrix.columns:
            print(f"⚠️ Warning: Wisata '{item}' tidak ditemukan dalam similarity_matrix. Melewati item ini.")
            continue  # Lewati wisata yang tidak ada dalam similarity matrix

        # Ambil daftar wisata yang mirip dengan wisata saat ini
        similar_items = similarity_matrix[item].drop(item, errors="ignore")  # Hindari kesamaan dengan dirinya sendiri

        # Ambil hanya wisata yang telah dikunjungi user
        available_items = user_ratings[user_ratings > 0].index

        # Cari irisan antara wisata mirip dan wisata yang dikunjungi user
        valid_items = similar_items.index.intersection(available_items)

        # Jika tidak ada wisata yang valid, lewati prediksi untuk item ini
        if valid_items.empty:
            # print(f"⚠️ Warning: Tidak ada wisata mirip yang sudah dikunjungi user untuk '{item}'. Melewati item ini.")
            continue

        # Pilih top-K wisata paling mirip yang sudah dikunjungi user
        similar_items = similar_items.loc[valid_items].sort_values(ascending=False).head(k)

        # Hitung prediksi rating menggunakan weighted sum
        numerator = sum(similar_items * user_ratings[similar_items.index])
        denominator = sum(similar_items.abs())

        # Pastikan tidak terjadi pembagian dengan nol
        predicted_rating = numerator / denominator if denominator != 0 else 0
        predictions[item] = predicted_rating

    # Buat DataFrame hasil prediksi untuk evaluasi
    result_df = pd.DataFrame({
        'Actual': user_ratings[rated_items],
        'Predicted': pd.Series(predictions, index=rated_items),
        'nama DTW': rated_items
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
    actual = predictions['Actual'].values
    predicted = predictions['Predicted'].values

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # Dalam persen

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


## semua user
def predict_for_all_users(rating_matrix, similarity_matrix, k=5):

    all_predictions = []  # Untuk menyimpan hasil prediksi setiap user

    for user_id in rating_matrix['reviewer'].unique():
        print(f"Prediksi untuk User: {user_id}")
        predictions_item = predict_existing_ratings(user_id, rating_matrix, similarity_matrix, k)
        all_predictions.append(predictions_item)

    # Gabungkan hasil prediksi untuk semua user
    final_predictions = pd.concat(all_predictions)

    return final_predictions