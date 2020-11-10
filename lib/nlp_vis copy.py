"""
@Andrew
Visualization Package for NLP Topic Modelling
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import cycle

def display_topics(model, feature_names, no_top_words, topic_names=None, show_weights=False):
    """
    Displays Top words associated with topics from Topic Modeling

    model: trained NLP Model (SVD, NMF)
    feature_names: feature names from vectorizers
    no_top_words: number of words to show
    topic_names: List of topic names to assign topics
    show_weights: True to show weights of important words. 
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        if show_weights:
            print([(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-no_top_words - 1:-1]])
        
        else:
            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

def plot_PCA_2D(data, target, target_names):
    """
    Plots the explained variance of two principal components of a multidimensional dataset. 

    data: PCA features to plot
    target: labels for targets
    target_names: Names to assign the clusters of colors
    """
    colors = cycle(['r','g','b','c','m','y','orange','w','aqua','yellow'])
    target_ids = range(len(target_names))
    plt.figure(figsize=(10,10))
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
                   c=c, label=label, edgecolors='gray')
    plt.legend(bbox_to_anchor=(1,1))

def plot_tSNE_2D(data, target, target_names):
    """
    Plots the TSNE visualization of high dimesnsional data
    data: TSNE results with 2 components
    target: labels for targets
    target_names: Names to assign the clusters of colors
    """
    x = data[:,0]
    y = data[:,1]
    plt.figure(figsize=(20,14))
    sns.scatterplot(
        x=x,
        y=y,
        hue=target, 
    )

def show_CorEx_topics(model):
    """
    Shows the words associated with the topics in a CoRex Model
    If not anchored, the topic correlation will be sorted in descending order
    """
    topics = model.get_topics()
    for n, topic in enumerate(topics):
        topic_words, _ = zip(*topic) 
        print(f'Topic {n}: TC Score:{model.tcs[n]}: \n', ', '.join(topic_words))
