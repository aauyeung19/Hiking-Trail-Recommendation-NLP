"""
@Andrew
"""

import pandas as pd
import numpy as np
import sqlalchemy
import psycopg2
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import re
import pickle

sp = spacy.load('en_core_web_sm')
stopwords_list = stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')
# tokenize by words 
# remove stop words
def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords_list]
    return ' '.join(filtered_tokens)

# remove numbers and punctuation
def remove_digits(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text
def lemma(text):
    text = sp(text)
    text = [word.lemma_ for word in text]
    text = ' '.join(text)
    return text

def clean_corpus(corpus):
    cleaned_corpus = []

    for doc in corpus:
        # lower case
        doc = doc.lower()
        # remove numbers and punctuation
        doc = remove_digits(doc)
        # remove stop words
        doc = remove_stopwords(doc)
        # lemmatize
        doc = lemma(doc)

        cleaned_corpus.append(doc)
    
    return cleaned_corpus

if __name__ == "__main__":
    conn=psycopg2.connect(database='alltrails', user='postgres', host='127.0.0.1', port= '5432')
    q = "SELECT hike_id, trail_description FROM hikes;"
    hikes_df = pd.read_sql_query(q, conn)
    hikes_df['cleaned_descriptions'] = clean_corpus(hikes_df['trail_description'])
    with open('../src/cleaned_hike_desc.pickle', 'wb') as to_write:
        pickle.dump(hikes_df, to_write)