# Word2Vec model visualization
"""
Created on Wed Nov 14 12:36:25 2018

@author: Rahul
"""


import nltk
import urllib
import bs4 as bs
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
nltk.download('stopwords')

# Data source
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/World_Diabetes_Day').read()

#  create BeautifulSoup object
soup = bs.BeautifulSoup(source,'lxml')

# Fetch the data
text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

# Preprocessing of data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Prepare the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec Model
model = Word2Vec(sentences, min_count=1)

words = model.wv.vocab

# Get the Word Vectors
vector = model.wv['global']

# Find similar words
similar = model.wv.most_similar('global')
