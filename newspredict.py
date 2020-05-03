'''"""
PROBLEM STATEMENT:-
    To build a model to accurately classify a piece of news as REAL or FAKE. Using sklearn,  build a TfidfVectorizer
    on the provided dataset. Then, initialize a PassiveAggressive Classifier and fit the model. In the end,
    the accuracy score and the confusion matrix tell us how well our model fares. On completion, create a GitHub
    account and create a repository. Commit your python code inside the newly created repository.


 Author = Vedant Deshpande 

 References = Medium.com
 '''
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix



dataset = pd.read_csv('news.csv')

print(dataset.info())

labels = dataset.label
x_train,x_test,y_train,y_test=train_test_split(dataset['text'], dataset['label'], test_size=0.2, random_state=7)

vectoriser = TfidfVectorizer()

tfidf_train = vectoriser.fit_transform(x_train)

tfidf_test = vectoriser.transform(x_test)

pac = KNeighborsClassifier(n_neighbors=3)

pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)

score = 	accuracy_score(y_test,y_pred)

print('accuracy = ', round(score*100,2))

print(confusion_matrix(y_test,y_pred,labels=['FAKE','REAL']))





