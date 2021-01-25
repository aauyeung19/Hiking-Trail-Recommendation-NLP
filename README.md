# Hiking Trail Recommendation App
## Abstract
In this time of social distancing, people are looking for different and new ways of safely getting outdoors.  The AllTrails app is a great way to find new trails and paths near you.  I never leave home on a trip without planning ahead to know what trail lies ahead.  In this project, I generated recommendations for my next adventure using trail descriptions and reviews offered on the platform. You can interact with the app [here](https://trailrecommendation.herokuapp.com/).

## Methodology
To generate recommendations, I had to go through three steps.  
1. Process and Clean the Data to prepare the data for step 2.  This included using a cleaning script that could be applied onto the entire corpus.  Aside from standard cleaning procedures like removing punctuation and numbers, I used spaCy to lemmatize, remove pronouns, and remove location names from the trail reviews. 
2. Performed Topic Modeling to group descriptions and reviews into topics.  This was done in two processes.  While NMF was a good solution for parsing trail descriptions and identifying key features, I needed something that would allow me to identify key elements in the reviews.  For that I chose to go with an anchored corex model so I could identify painpoints (if any) of each trail.
3. Generate Recommendation based on similarity in reviews and hike descriptions.  Something I found when cleaning the trail reviews was that people are most likely to only leave a review on 1 to 2 trails with very little overlap between users.  This made it difficult to perform collaborative filtering.  With that in mind, I opted for a content based filtering system that utilized two tables.  After the user selects the hike they enjoyed,  the app searches for other trails that had similar reviews (ex: If many people left reviews discussing the wildlife or types of views, hikes also discussing those topics would be returned).  Then, the app will sort that filtered list of hikes and return trails with similar trail descriptions.  This would act as a word of mouth recommendation system.  If many people are saying the same thing about two independent hikes, they must be similar! This way, the recommendation would be able to return hikes that are have similar reviews and similar features.  

## Data:
14,000 Trail Descriptions with 400,000 Reviews and 2,000,000 Ratings from AllTrails.com
Note: The review and ratings csv is too large to upload to GitHub.  If you would like to replicate the results please contact me @ andrew.k.auyeung@gmail.com. 

## Technologies:
* Sklearn: NMF, LDA, SVD
* NLP Technologies: NLTK, spaCy, CorEx
* Streamlit: Deployment of Application 

## Summary
With the recommendation system built, I will have to wait until I have a chance to try out the hikes before I can give it my full backing!
It sure does pass the eye test. 
*Note: The data for the app only covers certain states due to time, I was not able to scrape all 50 states.  This will be fixed in future iterations*
