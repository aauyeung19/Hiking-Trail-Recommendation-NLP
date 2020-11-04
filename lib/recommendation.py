"""
@Andrew

Comparrisons of two recommendation systems:
System 1: Collaborative Filtering using User Ratings
System 2: Determining distance metrics based on trail descriptions from NMF
+ Take one hike and search all hikes to find closest
--- Data is already transformed
--- search to fit conditions
--- Return 5 closest with either cosine or euclidean distance?
+ Take one hike and use NMF model to classify all the closest with the certain condition
--- Filter database on conditions set i.e. I want to stay in NY, I want to stay in NJ, I want to stay in this park
--- Transform data with NMF 
--- Return 5 closest with either cosine or euclidean distance?
--- OR --- KMeansClusters to determine similar hikes.  I.e Two hikes may be good with waterfalls and good with flowers
"""
from nlp import NLPPipe
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

# Method for determining Similar hikes
def compare_hikes_by_desc(pipe, hikes_df, hike_id, n_hikes_to_show):
    """
    Searches the hikes_df for hikes that are similar by cosine distance to the given hike_id. 
    Limits the search to the n_hikes_to_show
    args:
        pipe (NLPPipe): pipeline object
        hikes_df (dataframe): dataframe containing all the hike information
        hike_dtm (dataframe): Document Term Matrix of the hikes_df
        n_hikes_to_show (int): Number of relevant hikes to show
    returns:
        Filtered Dataframe with hikes that are close in cosine distance to the given hike. 
    """
    hikes_dtm = pipe.vectorizer.transform(hikes_df['cleaned_descriptions'])
    hikes_df, topics = pipe.topic_transform_df(hikes_df, hikes_dtm, append_max=False)
    X = hikes_df[topics]
    y = hikes_df.loc[[hike_id]][topics]
    similar_hike_idx = np.argsort(cosine_distances(X, y), axis=0)[:n_hikes_to_show+1]
    return hikes_df.iloc[similar_hike_idx.reshape(1,-1)[0]]

# Write method for filtering down by State/Park
def filter_hikes(hikes_df, states=None, parks=None, max_len=None, min_len=0):
    all_masks = []
    if states:
        mask = hikes_df['state'].isin(states)
        all_masks.append(mask)
    if parks:
        mask = hikes_df['park'].isin(parks)
        all_masks.append(mask)
    if max_len:
        mask = hikes_df['trail_length'] < max_len
        all_masks.append(mask)
    if min_len:
        mask = hikes_df['trail_length'] > min_len
        all_masks.append(mask)
    filtered_hikes = hikes_df[sum(all_masks).astype(bool)]
    
    return filtered_hikes

def user_recommendation(id, reviews):
    """
    id - selected id to compare
    reviews - filtered review dataframe 
    """
    r_ = reviews[['hike_id', 'user_id', 'user_rating']]
    r_.pivot_table(columns='hike_id', index='user_id', values='user_rating')
    user = r_[id]
    r_.corrwith(user)

if __name__ == "__main__":
    hikes_df = pd.read_csv('../src/clean_all_hikes.csv')
    # hikes_df = pickle.load(open('../src/cleaned_hike_desc.pickle', 'rb'))
    hikes_df.set_index('hike_id', inplace=True) 
    # Method for filtering conditions from DataFrame
    pipe = NLPPipe()
    pipe.load_pipe(filename='../models/nmf_trail_desc.mdl')

    # Filter First
    check = compare_hikes_by_desc(pipe, hikes_df, 'hike_219', 20)
