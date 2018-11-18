# -*- coding: utf-8 -*-
"""
@author: Rahul
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
import re

#getting the data 
reviewdata=load_files('txt_sentoken/')
x,y=reviewdata.data,reviewdata.target

#preprocessing of the data
corpuslist=[]
for i in range(0,2000):
    reviews=re.sub(r'\W',' ',str(x[i]))
    reviews=reviews.lower()
    reviews=re.sub(r'^br$',' ',reviews)
    reviews=re.sub(r'\s+br\s+',' ',reviews)
    reviews=re.sub(r'\s+[a-z]\s+',' ',reviews)
    reviews=re.sub(r'^b\s+',' ' ,reviews)
    reviews=re.sub(r'\s+',' ',reviews)
    corpuslist.append(reviews)
    
#Making a TFIDF Model    

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
x=vectorizer.fit_transform(corpuslist).toarray()

#Divide Data into Training andd Test Set    
from sklearn.model_selection import train_test_split
train_data, test_data, sent_train, sent_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

 # Train the classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(train_data,sent_train)


# Test model performance
sent_pred = classifier.predict(test_data)

#Evalute the Model Performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)