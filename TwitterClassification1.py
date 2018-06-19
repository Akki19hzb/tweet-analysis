import pandas as pd
import gensim
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from sklearn.decomposition import pca
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def MODEL_TEST(filename,label_encoder):
    df1 = pd.read_csv(filename, encoding="ISO-8859-1")
    x = df1['Tweets'].str.lower()
    X = []
    index = 0
    for tweet in x:
        # first we remove the links
        start_of_link = tweet.find('https') - 1
        tweet = tweet[0:start_of_link]

        # remove stop words
        tokens = word_tokenize(tweet)

        new_x = None
        for token in tokens:
            if token not in stop_words and token in model.wv.vocab:
                if new_x is None:
                    new_x = numpy.array(model[token])
                else:
                    new_x += numpy.array(model[token])
        if new_x is not None:
            X.append(new_x)
        # else:
            # del y[index]
        index += 1

    X = numpy.array(X)
    y=df1['Label']
    y = le.transform(y)
    y = list(y)
    # print(X)
    return X,y

model = Word2Vec.load('model.bin')
stop_words = set(stopwords.words('english'))

# remove https links as they are of no help and also @Airtel_Presence and all the stop words
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '@','airtel_presence',''])

df=pd.read_csv('Filtered_data_airtel.csv',encoding = "ISO-8859-1")

# make all the tweets in lower letters
x=df['Tweets'].str.lower()
y=df['Label']

# label encoder is used to convert the string names to appropriate numbers
le = LabelEncoder()

y=le.fit_transform(y)
y=list(y)

X=[]
index=0
for tweet in x:
    # first we remove the links
    start_of_link=tweet.find('https')-1
    tweet=tweet[0:start_of_link]

    # remove stop words
    tokens=word_tokenize(tweet)

    new_x=None
    for token in tokens:
        if token not in stop_words and token in model.wv.vocab:
            if new_x is None:
                new_x=numpy.array(model[token])
            else:
                new_x+=numpy.array(model[token])
    if new_x is not None:
        X.append(new_x)
    else:
        del y[index]
    index+=1

X=numpy.array(X)

y=numpy.array(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=1)
cls_model=RandomForestClassifier(n_estimators=10)
cls_model.fit(X_train,y_train)
y_pred=cls_model.predict(X_train)
print('Accuracy=%s'%accuracy_score(y_train,y_pred))

res_x,res_y=MODEL_TEST("abc.csv",le)
y_pred=cls_model.predict(res_x)
out=le.inverse_transform(y_pred)
out=list(out)
print(out,"\n")
print('Accuracy on new data from csv=%s' % accuracy_score(res_y, y_pred))











