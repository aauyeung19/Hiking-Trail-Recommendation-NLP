import streamlit as st
import pandas as pd
import numpy as np
import recommendation as rec 

@st.cache
def load_tables():
    hikes_df = pd.read_csv('../src/clean_all_hikes.csv', index_col=0)
    hikes_df.set_index('hike_id', inplace=True) 
    hikes_df = hikes_df[hikes_df['trail_length']<100]
    r_piv = pd.read_csv('../src/reviews_piv.csv', index_col=0)
    return hikes_df, r_piv


if __name__ == "__main__":
    hikes_df, r_piv = load_tables()

    st.title("All Trails Recommendations")

    st.write("What hike did you like?")
    st.write("Search list")
    state = st.selectbox('State', hikes_df["state"].unique())
    mask = (hikes_df["state"] == state)
    park = st.selectbox('Park', hikes_df[mask]['park'].unique())
    mask = mask & (hikes_df["park"] == park)
    trail_name = st.selectbox('Trail Name', hikes_df[mask]["trail_name"].unique())

    mask = mask & (hikes_df["trail_name"] == trail_name)
    
    st.write(
        "Trail Description"
    )
    st.write(
        hikes_df[mask]["trail_description"]
    )
    comp_id = hikes_df[mask].index

    st.write("Filter")
    state_f = st.multiselect('Filter the States you want to search', hikes_df["state"].unique())
    
    park_f = st.multiselect("Pick the Parks you want to checkout", hikes_df[hikes_df["state"].isin(state_f)]['park'].unique())
    
    len_bool = st.checkbox("Filter Length?", value=True)
    if len_bool:
        max_lim = rec.filter_hikes(hikes_df, states=state_f, parks=park_f)["trail_length"].max()
        min_lim = rec.filter_hikes(hikes_df, states=state_f, parks=park_f)["trail_length"].min()+1
        min_len = st.slider("Minimum Length (miles)", min_value=int(min_lim), max_value=int(max_lim))
        max_len = st.slider("Maximum Length (miles)", min_value=int(min_len)+2, max_value=int(max_lim))
    else:
        max_len, min_len = None, None


    filtered_df = rec.filter_hikes(hikes_df, states=state_f, parks=park_f, max_len=max_len, min_len=min_len)
    st.write(filtered_df.head())
    