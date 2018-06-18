import pandas as pd
import gensim
from gensim.models import Word2Vec
# define training data
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from sklearn.decomposition import pca
from matplotlib import pyplot



stop_words = set(stopwords.words('english'))

df=pd.read_csv('Filtered_data_airtel.csv',encoding = "ISO-8859-1")

# we need to make a corpus data to train the word2vec representation of words
# so we use the tweets and try to make a model.

# we first make all the text in lower case

df['Tweets']=df['Tweets'].str.lower()

# remove https links as they are of no help and also @Airtel_Presence and all the stop words
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '@','airtel_presence',''])

sentences=[]
for tweet in df['Tweets']:
    # first we remove the links
    start_of_link=tweet.find('https')-1
    tweet=tweet[0:start_of_link]

    # remove stop words
    tokens=word_tokenize(tweet)

    new_tweet=[]
    for token in tokens:
        if token not in stop_words:
            new_tweet.append(token)
    sentences.append(new_tweet)


# train model
model = Word2Vec(sentences,size=10, min_count=1)

# summarize vocabulary
words = list(model.wv.vocab)


# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
