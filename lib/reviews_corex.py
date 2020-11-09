import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import re
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
stop_words = stopwords.words('english')

def show_topics(model):
    """
    Shows the words associated with the topics in a CoRex Model
    If not anchored, the topic correlation will be sorted in descending order
    """
    topics = model.get_topics()
    for n, topic in enumerate(topics):
        topic_words, _ = zip(*topic) 
        print(f'Topic {n}: TC Score:{model.tcs[n]}: \n', ', '.join(topic_words))

r_ = pd.read_csv('../src/clean_reviews.csv', index_col=0)
for word in ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'black', 'white', 'violet']:
    stop_words.append(word)
r_.dropna(inplace=True)
vectorizer = CountVectorizer(stop_words='english', max_df=0.8, min_df=0.01)
r_dtm = vectorizer.fit_transform(r_['cleaned_reviews'])

# Get word names from vectorizer
words = list(vectorizer.get_feature_names())

# Train Topic Model
topic_model = ct.Corex(n_hidden=20, words=words, verbose=False)
topic_model.fit(r_dtm, words=words)

# Gets words from a single topic
topic_model.get_topics(topic=5, n_words=10)
show_topics(topic_model)

topic_model.p_y_given_x
topic_model.p_y_given_x.shape # Probability distribution of each document to the topic
topic_model.labels
topic_model.transform(r_dtm)
import pickle
with open('../models/corex_reviews.mdl', 'wb') as to_write:
    pickle.dump(topic_model, to_write)

# Retrain with Anchor Words

anchor_words = [['parking'], ['rock', 'steep'], ['family'], ['crowd']]
anchored_topic_model = ct.Corex(n_hidden=20)
anchored_topic_model.fit(r_dtm, words = words, anchors=anchor_words, anchor_strength=3)
show_topics(anchored_topic_model)

