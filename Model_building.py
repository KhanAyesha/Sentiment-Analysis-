# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:02:25 2021

@author: ayesha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

df = pd.read_csv(r'balanced_review.csv')
#df_s=pd.read_csv(r'D:\internship\scrappedReviews.csv')
df.shape

#data preparation
del df["summary"]

df.isna().sum()

#Review texts are missing.Doesn't make sense to replace them so we drop them 
df.dropna(inplace = True)

df.isna().sum()

df = df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )
df['Positivity'].value_counts()

#Data cleaning
#Normalization
def  clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df
data_clean = clean_text(df, 'reviewText', 'text_clean')
data_clean.head()

#Stop words
stop = stopwords.words('english')
data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_clean.head()

#Lemmatization
data_clean['text_tokens'] = data_clean['text_clean'].apply(lambda x: word_tokenize(x))
data_clean.head()
def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text
data_clean['text_tokens_lemma'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))
data_clean.head()

df['reviews']=df['text_tokens_lemma'].apply(lambda x: ' '.join(map(str, x)))


#Train test split
features_train, features_test, labels_train, labels_test = train_test_split(df['reviews'], df['Positivity'], random_state = 42 ) 

#bag of words
vect = CountVectorizer().fit(features_train)
features_train_vectorized = vect.transform(features_train)


model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))
roc_auc_score(labels_test, predictions)

#TF-IDF - term frequency inverse document frequency
#version 02

features_train, features_test, labels_train, labels_test = train_test_split(df['reviews'], df['Positivity'], random_state = 42 ) 
vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vectorized = vect.transform(features_train)

model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))
roc_auc_score(labels_test, predictions)

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
#save the count vectorizer
pickle.dump(vect.vocabulary_, open('feature.pkl', 'wb'))   
    




