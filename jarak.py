## Tools
import pandas as pd
import numpy as np
import re
import time
import random
import string
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.test.utils import datapath
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import torch.nn.functional as F
from Levenshtein import ratio
from sklearn.metrics import root_mean_squared_error
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Fungsi untuk menghitung jarak antar koordinat
def hitung_jarak(row):
    return geodesic(row["koordinat_dtw"], row["koordinat_user"]).kilometers

def preproses_koordinat(df):
  # buat fungsi untuk membaca koordinat
  def cek_koordinat(nama_string):
    # Generate 4 karakter acak (huruf dan angka)
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    # Pilih delay acak antara 1 hingga 5 detik
    delay = random.uniform(0.1, 1.5)
    time.sleep(delay)

    # Inisialisasi geolocator
    geolocator = Nominatim(user_agent=random_string)

    try:
      # Dapatkan lokasi
      lokasi = geolocator.geocode(nama_string)

      return lokasi.latitude, lokasi.longitude
    except:
      return 0,0

  # buat grouping berdasar "asal (user)"
  asal = df['asal (user)'].unique()
  df_asal = pd.DataFrame(asal, columns=['asal (user)'])
  # apply ke datase
  df_asal['koordinat_user'] = df_asal['asal (user)'].apply(cek_koordinat)
  # Gabungkan menjadi tuple (lat, lon)
  df["koordinat_dtw"] = df.apply(lambda row: f"({row.lattitude}, {row.longitude})", axis=1)
  # petakan koordinat di data df_asal ke dalam df menurut kolom asal(user)
  df = pd.merge(df, df_asal, on='asal (user)', how='left')

  # replace koordinat_user yang berisi (nan, nan); (°, °) atau ((blank), (blank)) menjadi nilai rata2
  df['koordinat_user'] = df['koordinat_user'].apply(lambda x: (df['koordinat_user'].mean(), df['koordinat_user'].mean()) if str(x) in ['(nan, nan)', '(°, °)', '((blank), (blank))'] else x)
  # Replace (nan, nan), (°, °), or ((blank), (blank)) with (0, 0)
  df['koordinat_dtw'] = df['koordinat_dtw'].apply( lambda x: (0, 0) if str(x) in ['(nan, nan)', '(°, °)', '((blank), (blank))'] else x)

  # Fungsi untuk mengonversi string ke tuple
  def convert_to_tuple(coord_str):
      # Jika data sudah berupa tuple, tidak perlu diproses lebih lanjut
      if isinstance(coord_str, tuple):
          return coord_str

      # Mengganti nilai tidak valid dengan (0, 0)
      if coord_str in ['(nan, nan)', '(°, °)', '((blank), (blank))']:
          return (0.0, 0.0)

      # Mengonversi string seperti '(-6.2088, 106.8456)' menjadi tuple (-6.2088, 106.8456)
      coord_str = coord_str.strip('()')  # Hilangkan tanda kurung
      lat, lon = coord_str.split(',')  # Pisahkan berdasarkan koma
      return (float(lat.replace('°', '').strip()), float(lon.replace('°', '').strip()))  # Konversi ke tuple (float, float)

  # Terapkan konversi ke kolom koordinat
  df['koordinat_dtw'] = df['koordinat_dtw'].apply(convert_to_tuple)
  df['koordinat_user'] = df['koordinat_user'].apply(convert_to_tuple)

  # hitung jarak
  df['jarak'] = df.apply(hitung_jarak, axis=1)
  return df

