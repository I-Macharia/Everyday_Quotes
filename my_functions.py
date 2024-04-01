import requests
import pandas as pd
from bs4 import BeautifulSoup
import scrapy 
import zipfile
#from pathlib import path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import FreqDist
import plotly.express as px

from langdetect import detect
from googletrans import Translator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def translate_to_english(text):
    """
Translates the given text to English if it is not already in English.

Args:
    text (str): The text to be translated.

Returns:
    str: The translated text in English, or the original text if it is already in English.

Example:
    ```python
    translated_text = translate_to_english("Bonjour")
    print(translated_text)  # Output: "Hello"
    ```
"""
    try:
        # Detect the language of the text
        lang = detect(text)

        if lang == 'en':
            return text
        translator = Translator()
        return translator.translate(text, src=lang, dest='en').text
    except:
        return text 
    

def preprocess_text(text):
    """
Preprocesses the given text by converting it to lowercase, tokenizing it, removing stopwords, and joining the tokens back into text.

Args:
    text (str): The text to be preprocessed.

Returns:
    str: The preprocessed text.

Example:
    ```python
    preprocessed_text = preprocess_text("This is a sample text.")
    print(preprocessed_text)  # Output: "sample text"
    ```
"""

    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

     # Tokenize
    words = nltk.word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    words = [word for word in words if word.isalpha() and word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join tokens back into text 
    
    return ' '.join(tokens), words


def compress_tags(tags_list):
    return list(set(tags_list))[:5]

def preprocesss_text(text):
    """
    Preprocesses the given text by converting it to lowercase, tokenizing it, removing stopwords, and joining the tokens back into text.
    
    Parameters:
    - text (str): The input text.
    
    Returns:
    - str: The preprocessed text.
    """
    if isinstance(text, float):
        # Return an empty string for float values
        return ''
    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Simple text cleaning process
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([c for c in text if c not in ('!', '.', ':', '?', ',', '\"')])
    return text

