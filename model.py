import pandas as pd
import numpy as np
from data import muat_data  # Asumsi modul ini sudah ada
from user_based_cf import *  # Asumsi modul ini sudah ada
from item_based_cf import *  # Asumsi modul ini sudah ada
from lda import *  # Asumsi modul ini sudah ada
from jarak import *
import os

# Tentukan folder untuk menyimpan file
folder = "./"  # Sesuaikan dengan direktori yang diinginkan

def rekomen(var_update=0, user="reviewer529"):
    """
    Fungsi untuk menghasilkan rekomendasi wisata berdasarkan user-based dan item-based CF.
    
    Parameters:
        var_update (int): 0 untuk data lama, 1 untuk kalkulasi baru
        user (str): Nama user
    
    Returns:
        pd.DataFrame: Tabel rekomendasi dengan kolom nama DTW, Predicted_Item, Predicted_User, Weighted_Average
    """
    
    if (var_update == 1) or ~(os.path.exists(folder + f"predictions_user_{user}.csv")):
        # Muat data
        df = muat_data()
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Gagal memuat data dari muat_data()")
        
    ###### Dijalankan jika variabel diupdate
    if var_update == 1:
        
        # Hapus kolom yang harusnya belum ada
        df = df.drop(["sentimen", "koordinat_dts", "koordinat_user", "jarak", "sentimen", "skor_cf", "predicted_topic", "predicted_topic_desc"], axis=1, errors='ignore')

        # Simpan data setelah hapus duplikat dan lowercase
        df.to_excel(folder + 'data_drop_duplicates.xlsx')

        # Hitung jarak
        df = preproses_koordinat(df)
        if df is None:
            raise ValueError("preproses_koordinat mengembalikan None")

        # Simpan data setelah dicari koordinat dan jarak
        df.to_excel(folder + 'data_plus_jarak.xlsx')

        # Cari sentimen review
        df = proses_sentimen(df)
        if df is None:
            raise ValueError("proses_sentimen mengembalikan None")

        # Simpan data setelah diprediksi sentimen
        df.to_excel(folder + 'data_plus_sentimen.xlsx')

        # Cari optimal topic
        optimal_topics_rev, model_list_rev = find_optimal_topics(2, 11, 1, df, 'review')
        optimal_topics_des, model_list_des = find_optimal_topics(2, 11, 1, df, 'deskripsi')

        # Simpan ke file
        with open(folder + "optimal_topics.txt", "w") as f:
            f.write(f"{optimal_topics_rev}\n")
            f.write(f"{optimal_topics_des}\n")

    # Baca dari file optimal_topics.txt
    if not os.path.exists(folder + "optimal_topics.txt"):
        raise FileNotFoundError(f"File {folder}optimal_topics.txt tidak ditemukan")
    with open(folder + "optimal_topics.txt", "r") as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError("File optimal_topics.txt tidak lengkap")
        optimal_topics_rev = int(lines[0].strip())  # Konversi ke integer
        optimal_topics_des = int(lines[1].strip())  # Konversi ke integer

    ###############################################################################

    ### USER BASED CF
    if (var_update == 1) or ~(os.path.exists(folder + f"predictions_user_{user}.csv")):
        # Parameter bobot w1, w2, w3
        w1 = 0.3  # rating
        w2 = 0.3  # sentimen
        w3 = 0.3  # jarak

        # Hitung skor cf
        df = hitung_skor_cf(df, w1, w2, w3)
        if df is None:
            raise ValueError("hitung_skor_cf mengembalikan None")

        # Buat matriks
        cf_user_df, cf_user_matriks = matriks_cf_user(df)
        if cf_user_df is None:
            raise ValueError("matriks_cf_user mengembalikan None")

        # Buat matriks lda ulasan
        lda_rev_df, lda_rev_matriks = lda_ulasan(df, optimal_topics_rev)
        if lda_rev_df is None:
            raise ValueError("lda_ulasan mengembalikan None")

        # Gabungkan kedua matriks similarity
        user_based_sim_matriks = (cf_user_df + lda_rev_df) / 2

        # Bangun matriks collaborative filtering berbasis user
        user_based_sim_df, _ = matriks_cf_user(df)

        # Pastikan tidak ada duplikat entri sebelum pivot
        df_grouped = df[['reviewer', 'nama DTW', 'skor_cf']].groupby(['reviewer', 'nama DTW']).mean().reset_index()

        # Buat pivot table
        rating_matrix = df_grouped.pivot(index='reviewer', columns='nama DTW', values='skor_cf')

        # Panggil fungsi prediksi dengan lebih dari 1 user yang mirip
        predictions_user = predict_existing_ratings_user_based(user, rating_matrix, user_based_sim_df)
        if predictions_user is None:
            raise ValueError("predict_existing_ratings_user_based mengembalikan None")

        # Simpan hasil prediksi ke dalam CSV
        predictions_user.to_csv(folder + f"predictions_user_{user}.csv", index=False)

    # Membaca kembali data prediksi dari CSV
    user_csv_path = folder + f"predictions_user_{user}.csv"
    if not os.path.exists(user_csv_path):
        raise FileNotFoundError(f"File {user_csv_path} tidak ditemukan")
    predictions_user = pd.read_csv(user_csv_path)

    ############################################## 

    ## ITEM BASED CF
    if (var_update == 1) or ~(os.path.exists(folder + f"predictions_user_{user}.csv")):
        # lda_ulasan_item
        ulasan_item_df = lda_ulasan_item(df, optimal_topics_rev)
        if ulasan_item_df is None:
            raise ValueError("lda_ulasan_item mengembalikan None")

        # lda_ulasan_item_des
        ulasan_item_des_df = lda_ulasan_item_des(df, optimal_topics_des)
        if ulasan_item_des_df is None:
            raise ValueError("lda_ulasan_item_des mengembalikan None")

        # Gabungkan lda ulasan, deskripsi, dan rating wisata
        df_rating = df[['nama DTW', "bintang destinasi"]].groupby('nama DTW').mean().reset_index()
        join_table = ulasan_item_df.merge(ulasan_item_des_df, on='nama DTW', how='inner').merge(df_rating, on='nama DTW', how='inner')

        # Matriks similarity
        item_based_sim_df, item_based_sim_matriks = item_based_similarity(join_table)
        if item_based_sim_df is None:
            raise ValueError("item_based_similarity mengembalikan None")

        predictions_item = predict_existing_ratings(user, df, item_based_sim_df, k=2)
        if predictions_item is None:
            raise ValueError("predict_existing_ratings mengembalikan None")

        # Simpan hasil prediksi ke dalam CSV
        predictions_item.to_csv(folder + f"predictions_item_{user}.csv", index=False)

    # Membaca data prediksi dari CSV
    item_csv_path = folder + f"predictions_item_{user}.csv"
    if not os.path.exists(item_csv_path):
        raise FileNotFoundError(f"File {item_csv_path} tidak ditemukan")
    predictions_item = pd.read_csv(item_csv_path)

    ##################################################

    # Gabungkan kedua DataFrame berdasarkan "nama DTW"
    df_new = predictions_item.join(predictions_user, lsuffix='_Item', rsuffix='_User')

    # Bobot untuk item-based dan user-based
    w_item = 0.2
    w_user = 0.8

    # Hitung rata-rata berbobot
    df_new["Weighted_Average"] = (w_item * df_new["Predicted_Item"]) + (w_user * df_new["Predicted_User"])

    # Sorting berdasarkan Weighted_Average dari yang terbesar
    df_sorted = df_new.sort_values(by="Weighted_Average", ascending=False)

    # Kembalikan 10 baris teratas sebagai hasil
    result = df_sorted[["nama DTW", "Predicted_Item", "Predicted_User", "Weighted_Average"]].head(10)
    if result.empty:
        raise ValueError("Hasil rekomendasi kosong")
    return result


# rekomen(var_update=1)