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

# Fungsi untuk menghapus URL
def remove_URL(tweet):
  if tweet is not None and isinstance(tweet, str):
      url = re.compile(r'https?://\S+|www\.\S+')
      return url.sub(r'', tweet)
  else:
      return tweet

def replace_words(text, mapping):
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, mapping.keys())) + r')\b')
    return pattern.sub(lambda x: mapping[x.group()], text)

# Fungsi untuk menghapus HTML
def remove_html(tweet):
  if tweet is not None and isinstance(tweet, str):
      html = re.compile(r'<.*?>')
      return html.sub(r'', tweet)
  else:
      return tweet

# Fungsi untuk menghapus emoji
def remove_emoji(tweet):
  if tweet is not None and isinstance(tweet, str):
      emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F700-\U0001F77F"  # alchemical symbols
          u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
          u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
          u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
          u"\U0001FA00-\U0001FA6F"  # Chess Symbols
          u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
          u"\U0001F004-\U0001F0CF"  # Additional emoticons
          u"\U0001F1E0-\U0001F1FF"  # flags
                              "]+", flags=re.UNICODE)
      return emoji_pattern.sub(r'', tweet)
  else:
      return tweet

# Fungsi untuk menghapus simbol
def remove_symbols(tweet):
  if tweet is not None and isinstance(tweet, str):
      tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)  # Menghapus semua simbol
  return tweet

# Fungsi untuk menghapus angka
def remove_numbers(tweet):
  if tweet is not None and isinstance(tweet, str):
      tweet = re.sub(r'\d', '', tweet)  # Menghapus semua angka
  return tweet

def remove_username(text):

  return re.sub(r'@[^\s]+', '', text)

def case_folding(text):
  if isinstance(text, str):
      lowercase_text = text.lower()
      return lowercase_text
  else:
      return text

def cleansing_data(text):
  text = case_folding(text)
  text = remove_URL(text)
  text = remove_html(text)
  text = remove_emoji(text)
  text = remove_symbols(text)
  text = remove_numbers(text)
  text = remove_username(text)
  return text

# BERT-CNN Model for Sentiment Analysis
class BERT_CNN(nn.Module):
    def __init__(self, num_classes=2, bert_model_name='indobenchmark/indobert-base-p1'):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Convolution layer
        self.conv = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)

        self.fc = None

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = bert_output.last_hidden_state

        # print("Shape after BERT:", embeddings.shape)

        embeddings = embeddings.permute(0, 2, 1)
        print("Shape after transpose:", embeddings.shape)

        conv_output = self.conv(embeddings)
        # print("Shape after Conv1d:", conv_output.shape)
        conv_output = self.relu(conv_output)
        pooled_output = self.pool(conv_output)
        # print("Shape after MaxPool1d:", pooled_output.shape)

        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        # print("Shape after flattening:", pooled_output.shape)
        if self.fc is None:
            self.fc = nn.Linear(pooled_output.size(1), 2).to(pooled_output.device)


        output = self.fc(pooled_output)
        return output

    def build():
        model_path = folder+'BERT-CNN_model.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sentiment_model = BERT_CNN(num_classes=2).to(device)
        sentiment_model.fc = nn.Linear(32768, 2).to(device)
        state_dict = torch.load(model_path, weights_only=True)
        sentiment_model.load_state_dict(state_dict)

        sentiment_model.eval()
        return sentiment_model


def predict_text_with_score(model, tokenizer, text, device, max_len=128):
    # Put the model in evaluation mode
    model.eval()

    # Preprocess the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Extract input IDs and attention mask
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        # For BERT models, outputs might be logits
        if hasattr(output, 'logits'):  # For compatibility with models like vanilla BERT
            output = output.logits

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the predicted label and its probability
        predicted_label = torch.argmax(probabilities, dim=1).item()
        sentiment_score = probabilities.squeeze().tolist()  # Convert probabilities to a list

    return predicted_label, round(5*sentiment_score[1],2)


def proses_sentimen(df):
  sentiment_model = BERT_CNN.build()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
  texts = df["review"].tolist()
  scores = []
  for text in texts:
    if not pd.isna(text):
      predicted_label, sentiment_scores = predict_text_with_score(sentiment_model, tokenizer, text, device)
      scores.append(sentiment_scores)
    else:
      scores.append(5)

  df.loc[:, 'sentimen'] = scores

  return df
