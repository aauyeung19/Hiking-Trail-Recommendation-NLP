"""
@Andrew

Methods in this file are used to tag reviews using their cosine similarity
to named topic keywords
"""

import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
sp = spacy.load('en_core_web_md')


########## 
# Think about making multiple tags
# Make a pipeline for feeding in topic labels and keywords 
# Have this pipeline return the tags for an associated review
# Aggregate the tags via voting from the reviews for a trail
# Append them to the tags list on hikes_df
##########

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

def tag_reviews_spacy(topics, topic_keywords, text, sp=sp):
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


if __name__ == "__main__":

    topic_labels = [
        'water',
        'floral',
        'mountain',
        'forest',
        'clean',
        'danger',
        'busy',
        'rocky',
        'family',
        'unmaintained',
        'buggy'
    ]
    topic_kws = [
        'water waterfall river lake pond',
        'flowers wildflowers',
        'overlook view mountain top ridge cliff lookout',
        'wood forest tree brush log',
        'well marked maintained clean',
        'dangerous careful',
        'full parking lot busy crowd',
        'boulder rocky scramble stair',
        'family young kid son daughter parent mom dad',
        'overgrown brush thick hard',
        'bug gnat mosquito sticky humid'
    ]

"""
Review Tags:
- Popularity: Keywords: busy crowded congested lots no parking 
- terrain type: Rocky, Swampy, Paved, Hilly, Trail, flat, steep, stairs, steps
- difficulty:
+ easy
+ medium
+ hard 
- scenic ( overlook / mountain)  keywords: overlook mountain top
- scenic ( waterfall / river) keywords: waterfall stream river lake water 
- keywords: flowers floral pretty wildflowers 
- multi-use / Running Friendly keywords: family kid 
- forest keywords: forest tree evergreen 

"""