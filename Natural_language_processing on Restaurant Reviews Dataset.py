
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
y = dataset.iloc[:, 1].values
# Cleaning the reviews
clean_review = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()       # object of porter stemmer class
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    clean_review.append(review)

# Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1200)
x = cv.fit_transform(clean_review).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Fitting Logistic Regression Classifier to  training data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Prediction of  Test results
y_pred = classifier.predict(x_test)

#Evaluate the model performance with Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
print(cm) 
print(accuracy)
print(classification_report(y_test, y_pred))