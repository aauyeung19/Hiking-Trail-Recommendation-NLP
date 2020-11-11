from wordcloud import WordCloud
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

r_ = pd.read_csv('r_corex.csv', index_col=0)
hike_info = pd.read_csv('hikes_df.csv')
r_ = r_.merge(hike_info[['hike_id', 'state']], left_on='hike_id', right_on='hike_id')
del hike_info

# Join words together
words = r_.groupby('state')['cleaned_reviews'].apply(lambda x: ' '.join(x))

# New Jersey
wordcloud = WordCloud(background_color="white").generate(words['Washington'])
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('New York')
plt.axis("off")
plt.show()





for state, text in enumerate(words):
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(words[state])
    plt.axis("off")
    plt.show()
