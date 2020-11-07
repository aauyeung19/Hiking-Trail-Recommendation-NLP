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
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import numpy as np
from copy import deepcopy
from reviews_embeddings import get_embedded_vectors


def compare_by_cosine_distance(X, y, n_to_limit=None):
    """
    Helper Function to compare one vectorized component (y)
    to a matrix of vectors (X)
    Returns the top n_to_limit closest values in X
    """
    idx = np.argsort(cosine_distances(X, y), axis=0)[:n_to_limit]
    idx = idx.reshape(1,-1)[0]
    return idx

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
    if type(hike_id) == str:
        hike_id = [hike_id]
    # After everything is done we can change this to just save the transformed data in a CSV.
    hikes_dtm = pipe.vectorizer.transform(hikes_df['cleaned_descriptions'])
    hikes_df, topics = pipe.topic_transform_df(hikes_df, hikes_dtm, append_max=False)
    X = hikes_df[topics]
    y = hikes_df.loc[hike_id][topics]
    similar_hike_idx = compare_by_cosine_distance(X, y, n_hikes_to_show)
    return hikes_df.iloc[similar_hike_idx]

############# FOR OPTIMIZATION: SAVE VECTORS INTO THE DOCUMENT 
def compare_hikes_by_desc_vec(hikes_df, hike_id, n_hikes_to_show):
    """
    Compares hike Descriptions based on the vectorized forms of their 
    words. 
    args:
        hikes_df (dataframe): dataframe containing all the hike information
        hike_dtm (dataframe): Document Term Matrix of the hikes_df
        n_hikes_to_show (int): Number of results to limit hikes to
    returns:
        Filtered Dataframe with hikes that are close in cosine distance to the given hike. 
    """
    emb_vecs = get_embedded_vectors(hikes_df['cleaned_descriptions'])
    comp_vec = get_embedded_vectors([hikes_df.loc[hike_id, 'cleaned_descriptions']])
    similar_hike_idx = compare_by_cosine_distance(emb_vecs, comp_vec, n_hikes_to_show)
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
        mask = "(hikes_df['state'].isin(states))"
        all_masks.append(mask)
    if parks:
        mask = "(hikes_df['park'].isin(parks))"
        all_masks.append(mask)
    if max_len:
        mask = "(hikes_df['trail_length'] < max_len)"
        all_masks.append(mask)
    if min_len:
        mask = "(hikes_df['trail_length'] > min_len)"
        all_masks.append(mask)

    if all_masks:
        all_masks = " & ".join(all_masks)
        filtered_hikes = deepcopy(hikes_df[eval(all_masks)])
    else:
        print('Warning! No Filters Applied to Dataframe')
        filtered_hikes = deepcopy(hikes_df)
    
    return filtered_hikes


def user_recommendation(comp_id, r_):
    """
    Collaborative Filtering for Recommendation
    comp_id - selected id to compare
    r_ - filtered review dataframe 
    """
    r_piv = r_.pivot_table(columns='hike_id', index='user_id', values='user_rating')
    user = r_piv.loc[:,comp_id]
    results = r_piv.corrwith(user)
    colab_top = results.dropna().sort_values(ascending=False).index
    if len(colab_top) == 1:
        return None
    else:
        return colab_top

if __name__ == "__main__":
    hikes_df = pd.read_csv('../src/clean_all_hikes.csv', index_col=0)
    # hikes_df = pickle.load(open('../src/cleaned_hike_desc.pickle', 'rb'))
    hikes_df.set_index('hike_id', inplace=True) 
    # Method for filtering conditions from DataFrame
    pipe = NLPPipe()
    pipe.load_pipe(filename='../models/nmf_trail_desc.mdl')

    # Select one hike from New Jersey for Comparision 
    nj = ['New Jersey']
    nj_hikes = filter_hikes(hikes_df, nj)
   # want to check specific hikes
    comp_id = np.random.choice(nj_hikes.index)


    # Filter by State for testing
    states = ['New York', 'Connecticut', 'Vermont', 'Maine',
       'Maryland', 'Colorado', 'Pennsylvania', 'New Hampshire',
       'Rhode Island', 'Massachusetts', 'Virginia', 'Tennessee',
       'North Carolina', 'Washington', 'West Virginia', 'Ohio',
       'South Carolina']
    filtered_hikes = filter_hikes(hikes_df, states)
    filtered_hikes = pd.concat([filtered_hikes, nj_hikes.loc[[comp_id]]])

    # Get 20 hikes by topic cosine closeness
    similar_topics = compare_hikes_by_desc(pipe, filtered_hikes, comp_id, 20)

    # Embedded top 20
    similar_vecs = compare_hikes_by_desc_vec(filtered_hikes, comp_id, 20)
    
    # load reviews
    reviews_df = pd.read_csv('../src/clean_reviews.csv', index_col=0)
    reviews_df.set_index('hike_id', inplace=True)

    # Filter Reviews down to subset of the top 20 closest reviews
    r_ = reviews_df.loc[similar_topics.index]

    collab_idx = user_recommendation(comp_id, r_)
    collab_hikes = hikes_df.loc[collab_idx]

    hikes_df.loc['hike_8002']['link']
    r_piv

    ### COMBINATION OF REVIEW AND DESCRIPTION TOPICS
    r_piv = pd.read_csv('../src/reviews_piv.csv', index_col=0)
    X = r_piv

    comp_ids = ['hike_115', 'hike_3204', 'hike_13002']
    y = r_piv.loc[comp_ids]
    h_idx = X.iloc[compare_by_cosine_distance(X, y, 10)].index
    a = hikes_df.loc[h_idx]
    compare_hikes_by_desc(pipe, a, comp_ids, n_hikes_to_show=2)