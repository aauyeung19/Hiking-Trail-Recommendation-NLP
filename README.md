# Hiking Trail 
## Abstract

In this time of social distancing, people are looking for different and new ways of safely getting outdoors.  The AllTrails app is a great way to find new trails and paths near you.  I never leave home on a trip without planning ahead to know what trail lies ahead.  In this project, I generated recommendations for my next adventure using trail descriptions and reviews offered on the platform. You can interact with the app [here](https://trailrecommendation.herokuapp.com/).


## Methodology
To generate recommendations, I had to go through three steps.  
1. Process and Clean the Data to prepare the data for step 2
2. Perform Topic Modelling to group descriptions and reviews into topics
3. Generate Recommendation based on similarity in reviews and hike descriptions 

## Data:
14,000 Trail Descriptions with 400,000 Reviews and 2,000,000 Ratings from AllTrails.com
Note: The review and ratings csv is too large to upload to GitHub.  If you would like to replicate the results please contact me @ andrew.k.auyeung@gmail.com. 

## Technologies:
Sklearn: NMF, LDA, SVD
NLP Technologies: NLTK, spaCy, CorEx
Streamlit: Deployment of Application 

## Summary
With the recommendation system built, I will have to wait until I have a chance to try out the hikes before I can give it my full backing!
It sure does pass the eye test. 
https://trailrecommendation.herokuapp.com/
*Note: The data for the app only covers certain states due to time, I was not able to scrape all 50 states.  This will be fixed in future iterations*
