"""
@Andrew

Methods in this file are used to tag reviews using their cosine similarity
to named topic keywords
"""
import pandas as pd
import copy
import seaborn as sns

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
sp = spacy.load('en_core_web_md')



topic_labels = [
    'water',
    'floral',
    'forest',
    'clean',
    'danger',
    'rocky',
    'family',
    'unmaintained',
    'buggy',
    'dog',
    'photography',
    'flat',
    'camping',
    'busy'
]
topic_kws = [
    'water waterfall river lake',
    'flowers wildflowers',
    'wood forest tree brush log',
    'marked maintained clean',
    'dangerous scary',
    'boulder rocky scramble stair',
    'family young',
    'overgrown brush thick hard',
    'bug gnat mosquito sticky humid',
    'dog',
    'photo camera photoshoot',
    'flat',
    'camping backpack tent',
    'crowded crowd'
]

def get_embedded_vectors(words, sp=sp):
    """
    helper function for tag_reviews_spacy
    returns average word vectors of the words in a string
    """
    # turn words and text to spacy objects
    word_docs = list(sp.pipe(words))
    word_vectors = []
    for doc in word_docs:
        if doc.has_vector:
            # converts each word in word keywords into a vector
            # appends the average vector 
            word_vectors.append(doc.vector)
        else:
            # if the word does not exist in the library, return length 300 list of zeroes
            sp.vocab[0].vector
    # return as np array
    word_vectors = np.array(word_vectors)
    return word_vectors

def tag_reviews_spacy(text, topics=topic_labels, topic_keywords=topic_kws, sp=sp):
    """
    Searches text and returns labeled topics using given keywords
    with similar cosine distances.
    args:
        topics (list): list of topics 
        topic_keywords (list): list of keywords. \
            Each item in this list should match \
            with its associated topic label in *topics*
        text (array): the cleaned corpus to tag
        sp (spacy): loaded nlp library
    returns:
        text_topics (array): list with tags
    """
    if len(topics) != len(topic_keywords):
        raise ValueError('Length of Topics and Topic Keywords should match!')
    
    ####### Consider putting this outside the function so it doesn't run for every row
    topic_vectors = get_embedded_vectors(words=topic_keywords, sp=sp)
    # This still needs to run for every row
    text_vectors = get_embedded_vectors(words=text, sp=sp)
    
    similarity = cosine_similarity(text_vectors, topic_vectors)
    topic_idx = np.argmax(similarity, axis=1)
    text_topics = [topics[i] for i in topic_idx]
    return text_topics

def pivot_by_tags(df, review_col_name):
    """
    Pivot counts of tags by hike and returns a pivot table.
    Uses tag_reviews_spacy() method to generate tag counts. 
    """
    df['review_tags'] = tag_reviews_spacy(text=df[review_col_name], topics=topic_labels, topic_keywords=topic_kws)
    df_piv = df.groupby(['hike_id', 'review_tags'], as_index=False)[review_col_name].count() \
        .pivot_table(index='hike_id', columns='review_tags', values=review_col_name)
    return df_piv, df

if __name__ == "__main__":

    reviews_df = pd.read_csv('../src/clean_reviews.csv', index_col=0)
    reviews_df.set_index('hike_id', inplace=True)
    reviews_df.dropna(inplace=True)
    reviews_df['cleaned_reviews'] = reviews_df['cleaned_reviews'].map(str)
    
    stop_words = ['great', 'good', 'fun', 'nice', 'easy', 'love', 'awesome']
    reviews_df['cleaned_reviews'] = reviews_df['cleaned_reviews'].map(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))


    # np.random.choice(reviews_df.index)
    r_ = copy.deepcopy(reviews_df.iloc[:10000])
    tags = tag_reviews_spacy(text=r_['cleaned_reviews'], topics=topic_labels, topic_keywords=topic_kws)
    r_['review_tags'] = tags
    sns.countplot(y=tags, orient='h')
    print(r_.index.unique()[0])
    
    temp = r_[['user_desc', 'review_tags']]
    # SAME thing as Count Vectorizer
    # Creates pivot table
    r_.reset_index(inplace=True)
    r_piv = r_.groupby(["hike_id", "review_tags"], as_index=False)["user_id"].count() \
        .pivot_table(index="hike_id", columns="review_tags", values="user_id")
    # Recode above for TFIDF 
    from sklearn.metrics import pairwise_distances
    # standardize columns
    r_piv = r_piv.sub(r_piv.mean(axis=1), axis=0).div(r_piv.std(axis=1), axis=0)
    r_piv.fillna(0, inplace=True)
    distances = pd.DataFrame(pairwise_distances(r_piv, metric='cosine'), index=r_piv.index, columns=r_piv.index)
    distances.sort_values(by='hike_13')['hike_13']
    r_piv.sort_values(by=['camping', 'not busy', 'water'], ascending=False)


    ###############################################
    # Embedded Vectors for sentences
    ###############################################
    r_sent = pd.read_csv("../src/reviews_by_sent.csv", index_col=0)
    r_sent.dropna(inplace=True)
    stop_words = ['great', 'good', 'fun', 'nice', 'easy', 'love', 'awesome', '-PRON-', 'trail']
    r_sent['clean_review'] =r_sent['clean_review'].map(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))
    r_piv, r_sent = pivot_by_tags(r_sent, 'clean_review')
    r_piv.fillna(0, inplace=True)
    r_piv = r_piv.sub(r_piv.mean(axis=1), axis=0).div(r_piv.std(axis=1), axis=0)
    r_piv.to_csv('../src/reviews_piv.csv')
    sns.heatmap(r_piv, center=0)

