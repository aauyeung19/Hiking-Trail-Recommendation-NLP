import streamlit as st
import pandas as pd
import numpy as np
from recommendation import comparrison, filter_hikes 

@st.cache
def load_tables():
    hikes_df = pd.read_csv('hikes_df.csv', index_col=0)
    # hikes_df = hikes_df[hikes_df['trail_length']<100]
    hike_tag_dummies = pd.read_csv('hike_tag_dummies.csv', index_col=0)
    ht_mat = pd.read_csv('ht_mat.csv', index_col=0)
    dt_mat = pd.read_csv('dt_mat.csv', index_col=0)
    reviews = pd.read_csv('r_corex.csv', index_col=0)
    return hikes_df, hike_tag_dummies, ht_mat, dt_mat, reviews

@st.cache(allow_output_mutation=True)
def store_random_id():
    return []

@st.cache(allow_output_mutation=True)
def store_comp_id():
    return []



    


if __name__ == "__main__":
    hikes_df, hike_tag_dummies, ht_mat, dt_mat, reviews = load_tables()
    # st.image('header.JPG', use_column_width=True)
    st.title("All Trails Recommendations")
    method = st.radio('Pick your method', ["Select a Hike", "I'm Feeling Lucky"])
    
    if method == "Select a Hike":
        st.write("What hike did you like?")
        state = st.selectbox('State', hikes_df["state"].unique())
        mask = (hikes_df["state"] == state)
        park = st.selectbox('Park', hikes_df[mask]['park'].unique())
        mask = mask & (hikes_df["park"] == park)
        trail_name = st.selectbox('Trail Name', hikes_df[mask]["trail_name"].unique())
        mask = mask & (hikes_df["trail_name"] == trail_name)
        comp_id = hikes_df[mask].index

    else:
        cached_id = store_random_id()
        if len(cached_id) == 0:
            st.balloons()
            cached_id.append(np.random.choice(hikes_df.index))
        comp_id = hikes_df.loc[[cached_id[0]]].index

        clear_cache = st.button('Show me another', True)
        if clear_cache:
            cached_id.clear()
            cached_id.append(np.random.choice(hikes_df.index))
            comp_id = hikes_df.loc[[cached_id[0]]].index
            clear_cache = False

        st.subheader("Trail Name")
        st.write(hikes_df.loc[comp_id]["trail_name"][0])
        st.subheader("State")
        st.write(hikes_df.loc[comp_id]["state"][0])
        st.subheader("Park")
        st.write(hikes_df.loc[comp_id]["park"][0])

    # Show Selection Descriptions
    st.subheader("Trail Description")
    st.write(hikes_df.loc[comp_id]["trail_description"][0])
    st.write("Review Tags: ", hikes_df.loc[comp_id]["temp_tag"][0])

    # Show reviews that talk about tags
    filtered_reviews = reviews[reviews['hike_id']==comp_id[0]][['date','user_desc']]
    st.table(filtered_reviews.set_index('date'))

    st.header("Filter Your Hikes")
    state_filter = st.multiselect('State', hikes_df["state"].unique())
    mask = (hikes_df["state"].isin(state_filter))
    park_filter = st.multiselect('Park', hikes_df[mask]['park'].unique())
    min_len_filter = st.slider('Minimum Trail Length', 0, 400)
    max_len_filter = st.slider('Maximum Trail Length', min_len_filter+1, 400, 400)
    
    filtered_indexes = filter_hikes(
        hikes_df, 
        states=state_filter, 
        parks=park_filter,
        max_len=max_len_filter,
        min_len=min_len_filter).index
    filtered_indexes = filtered_indexes.append(comp_id)
    ht_mat = ht_mat.loc[filtered_indexes]
    dt_mat = dt_mat.loc[filtered_indexes]
    ht_mat.drop_duplicates(inplace=True)
    dt_mat.drop_duplicates(inplace=True)

    show = st.checkbox("Show Me Similar Hikes!", True)

    if not ht_mat.shape[0]:
        st.error('No Trails Exist with the Filtered Conditions.')
    
    if not show:    
        st.stop()

    sim_idx = comparrison(comp_id=comp_id[0], ht_mat=ht_mat, dt_mat=dt_mat, r_lim=30, desc_lim=3)
    if len(sim_idx) == 1:
        st.error('No Trails Exist with the Filtered Conditions.')
    for n, i in enumerate(sim_idx[1:], start=1):
        st.header(f"Recommendation {n}")
        st.subheader("Trail Name")
        st.write(hikes_df.loc[i]["trail_name"])
        st.subheader("State")
        st.write(hikes_df.loc[i]["state"])
        st.subheader("Park")
        st.write(hikes_df.loc[i]["park"])
        st.subheader("Trail Description")
        st.write(hikes_df.loc[i]["trail_description"])
        st.write("Review Tags: ", hikes_df.loc[i]["temp_tag"])
        st.write("Link: ", hikes_df.loc[i]["link"])
        st.markdown("-------------------------------------")
        