#!/usr/bin/python
# -*- coding: utf-8 -*- 

"""
Naive Bayes classifiers are a popular statistical technique of e-mail filtering. 
They typically use bag of words features to identify spam e-mail.

@date: Thu 5 Mar. 2020
"""

# Import necessary modules.
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib


# Read csv file in 'lation-1'.
df = pd.read_csv('spam.csv', encoding='latin-1')

# Remove extra columns.
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Convert label from string to 0/1.
df['v1'].map({'ham': 0, 'spam':1})
df.rename(columns={'v1': 'label', 'v2':'message'},inplace=True)

X = df['message']
y = df['label']

# Convert text to bag of words.
cv = CountVectorizer()
X = cv.fit_transform(X)

# Convert text to tf-idf.
# tv = TfidfVectorizer()
# X = tv.fit_tranform(X) 

# Split data into training & testing datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier.
clf = MultinomialNB()
clf.fit(X_train, y_train)		# Fit Naive Bayes classifier
clf.score(X_test, y_test)		# Return the mean accuracy

y_pred = clf.predict(X_test)	# Perform classification on testing data

print(classification_report(y_test, y_pred))

# Save model as a .pkl file.
joblib.dump(clf, 'NB_spam_model.pkl')

# Load and use saved model.
# NB_spam_model = open('NB_spam_model.pkl','rb')
# clf = joblib.load(NB_spam_model)
