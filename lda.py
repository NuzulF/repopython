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

def compute_coherence_values(dictionary, corpus, texts, start=2, limit=10, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values

def preprocess_text(text,stop_words):
    stop_words += ['menjadi','sangat','banget','mau','menjadi','jadi','buat','yg','jd','bs','dg','dgn','tp','km','k','sgt','nya','utk','cukup',
                   'masuk','kalo','kalau','banyak','tempat']
    result = [word for word in simple_preprocess(text) if word not in stop_words]
    return result

# Plot coherence scores
def construct_corpus(data,column):

  stop_factory = StopWordRemoverFactory()
  stop_words = stop_factory.get_stop_words()
  data[column] = data[column].astype(str)


  processed_texts = [preprocess_text(text,stop_words) for text in data[column]]
  dictionary = corpora.Dictionary(processed_texts)
  corpus = [dictionary.doc2bow(text) for text in processed_texts]
  return dictionary,corpus,processed_texts,stop_words


def find_optimal_topics(start, limit, step,data,column):

  dictionary,corpus,processed_texts,stop_words = construct_corpus(data,column)
  model_list, coherence_values = compute_coherence_values(dictionary, corpus, processed_texts, start, limit, step)
  x = range(start, limit, step)
  optimal_topics = x[coherence_values.index(max(coherence_values))]
  plt.plot(x, coherence_values)
  plt.xlabel("Number of Topics")
  plt.ylabel("Coherence Score")
  plt.title("Optimal Number of Topics")
  plt.show()
  print(f"Optimal number of topics: {optimal_topics}")
  return optimal_topics,model_list

def most_dominant_topic(lda_model,data,column, text):
  dictionary,corpus,processed_texts,stop_words = construct_corpus(data,column)
  new_doc_bow = dictionary.doc2bow(preprocess_text(text,stop_words))

  topic,score = max(lda_model.get_document_topics(new_doc_bow), key=lambda x: x[1])
  return topic,score