#def nilai evaluasi model item base (ht)?
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
    rated_items = user_ratings[user_ratings > 0].index

    predictions = {}
    for item in rated_items:
        if item not in similarity_matrix.columns:
            # print(f"⚠️ Warning: Wisata '{item}' tidak ditemukan dalam similarity_matrix. Melewati item ini.")
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
    actual = predictions['Actual'].values
    predicted = predictions['Predicted'].values

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100  # Dalam persen

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# # Contoh pemanggilan fungsi
# predictions_item = predict_existing_ratings("🇮🇩hartonoswanopati🇮🇩", df, item_based_sim_df, k=3)
# print(predictions_item)

# # Evaluasi performa prediksi
# metrics = evaluate_model(predictions_item)
# print("Evaluation Metrics:", metrics)
