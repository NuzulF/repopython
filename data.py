import pandas as pd
import numpy as np

def muat_data():
    # muat lagi data_wisata_final
    df = pd.read_csv('tabel_utama_anonim.csv')

    # hapus duplikasi berdasar nama DTW, reviewer dan review
    df = df.drop_duplicates(subset=['nama DTW', 'reviewer', 'review', 'rating'])

    # hapus kolom kosong
    df = df.drop([" .1","Kategori","sumber deskripsi tambahan","asal negara","kec (lokasi wisata)","kabupaten (lokasi wisata)"], axis=1)

    # lowercase semua data
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    df = df.copy()
    # Isi nilai kosong pada kolom numerik dengan mean (rata-rata)
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.mean()))

    # Isi nilai kosong pada kolom kategorikal dengan modus (nilai yang paling sering muncul)
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0] if not col.mode().empty else ""))

    return df