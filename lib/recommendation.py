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
from copy import deepcopy

# Method for determining Similar hikes
def compare_hikes_by_desc(pipe, hikes_df, hike_id, n_hikes_to_show):
    """
    Searches the hikes_df for hikes that are similar by cosine distance to the given hike_id. 
    args:
        pipe (NLPPipe): pipeline object
        hikes_df (dataframe): dataframe containing all the hike information
        hike_dtm (dataframe): Document Term Matrix of the hikes_df
        n_hikes_to_show (int): Number of results to limit hikes to
    returns:
        Filtered Dataframe with hikes that are close in cosine distance to the given hike. 
    """
    hikes_dtm = pipe.vectorizer.transform(hikes_df['cleaned_descriptions'])
    hikes_df, topics = pipe.topic_transform_df(hikes_df, hikes_dtm, append_max=False)
    X = hikes_df[topics]
    y = hikes_df.loc[[hike_id]][topics]
    similar_hike_idx = np.argsort(cosine_distances(X, y), axis=0)[:n_hikes_to_show]
    similar_hike_idx = similar_hike_idx.reshape(1,-1)[0]
    return hikes_df.iloc[similar_hike_idx]

# Write method for filtering down by State/Park
def filter_hikes(hikes_df, states=None, parks=None, max_len=None, min_len=0):
    """
    Filter Hikes function to shrink dataframe down before creating recommendations.  
    args:
        hikes_df (DataFrame): Cleaned Hikes DataFrame
        states (list): list of desired states
        parks (list): list of desired parks
        max_len (int): Longest length for hike
        min_len (int): Minimum length of hike
    return:
        filtered_hikes (DataFrame): hikes_dataframe filtered by the parameters
    """
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
    filtered_hikes = deepcopy(hikes_df[sum(all_masks).astype(bool)])
    
    return filtered_hikes


def user_recommendation(comp_id, reviews):
    """
    id - selected id to compare
    reviews - filtered review dataframe 
    """
    r_piv = r_.pivot_table(columns='hike_id', index='user_id', values='user_rating')
    user = r_piv.loc[:,comp_id]
    r_piv.corrwith(user)

if __name__ == "__main__":
    hikes_df = pd.read_csv('../src/clean_all_hikes.csv', index_col=0)
    # hikes_df = pickle.load(open('../src/cleaned_hike_desc.pickle', 'rb'))
    hikes_df.set_index('hike_id', inplace=True) 
    # Method for filtering conditions from DataFrame
    pipe = NLPPipe()
    pipe.load_pipe(filename='../models/nmf_trail_desc.mdl')

    # Filter by State for testing
    states = ['New Jersey', 'New York']
    filtered_hikes = filter_hikes(hikes_df, states)
    
    # want to check specific hikes
    comp_id = 'hike_5'

    # Get 20 hikes by topic cosine closeness
    check = compare_hikes_by_desc(pipe, filtered_hikes, comp_id, 20)

    # load reviews
    reviews_df = pd.read_csv('../src/clean_reviews.csv', index_col=0)
    reviews_df.set_index('hike_id', inplace=True)
    
    # Filter Reviews down to subset of the top 20 closest reviews
    r_ = reviews_df.loc[check.index]


    r_piv = r_.pivot_table(columns='hike_id', index='user_id', values='user_rating')
    user = r_piv.loc[:,comp_id]
    results = r_piv.corrwith(user)
    colab_top = results.dropna().sort_values(ascending=False).head(10).index
    hikes_df.loc[top_5]