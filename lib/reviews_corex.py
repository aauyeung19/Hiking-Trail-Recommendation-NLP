import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import re
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
from nlp import NLPPipe
from cleaning import clean_corpus
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

if __name__ == "__main__":
    
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


    # Retrain with Anchor Words
    reviews = pd.read_csv("../src/cleaned_reviews_5.csv", index_col=0)
    reviews.dropna(inplace=True)    
    vectorizer = CountVectorizer(stop_words='english', max_df=0.8, min_df=0.01, ngram_range=(1,2))
    anchor_words = [['parking', 'crowd'], ['rock', 'rocky'], ['ice', 'snow'], ['lake', 'waterfall', 'pond'], ['easy'], ['hard'], ['bug'], ['family'], ['maintain']]
    anchored_topic_model = ct.Corex(n_hidden=10)
    cleaning_function = clean_corpus

    pipe = NLPPipe(vectorizer=vectorizer, model=anchored_topic_model, cleaning_function=cleaning_function)
    r_dtm = pipe.vectorizer.fit_transform(reviews["cleaned_reviews"])
    words = list(pipe.vectorizer.get_feature_names())
    pipe.model.fit(r_dtm, words = words, anchors=anchor_words, anchor_strength=3)

    test = ['We went back in September. I’m bad at posting reviews, but honestly this trail is pretty rad. The drive up absolutely sucks. It’s super gravely and bumpy and parking is a nightmare. Besides that whole mess the hike is steep and the climb does suck, but it’s worth it. The views at the top are incredible. Even the lake at the bottom is a dream to stop and relax at. Definitely stop for snack breaks, but if you can make it to the fire lookout you should. The fire lookout was closed, but you can still see everything inside and the views around it were breathtaking. If you can endure the suck, you should do this hike. ']
    topics = np.array(['Parking Issues', 'Rocky Conditions',
       'Snow/Icy Conditions', 'Lake/Waterfall/Pond', 'Easy Difficulty',
       'Hard Difficulty', 'Bring Bug Spray', 'Family Friendly',
       'Well Maintained'])
    cleaned_text = pipe.cleaning_function(test)
    test_dtm = pipe.vectorizer.transform(cleaned_text)
    results = pipe.model.transform(test_dtm)
    np.ma.array(topics, mask=~results[0][:9])

    
    pipe.save_pipe('../models/review_corex.mdl')