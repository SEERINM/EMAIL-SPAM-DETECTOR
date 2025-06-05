import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
#count vectorizer - converrt text to numerical data
df = pd.read_csv('spam.csv', encoding='latin-1')
print(df.columns)
df = df[['v1','v2']]
df.columns=['category','message']
#print(df.head())
#rows and cols
#print(df.shape)
#preprocessing the data
df.drop_duplicates(inplace=True)
#print(df.shape)
#print(df.isnull().sum())
#replacing ham with not spam
df['category'] = df['category'].replace(['ham','spam'],['Not Spam','Spam'])
print(df.head(3))
mes = df['message']
cat = df['category']
#spliting the data as test and train
(x_train,x_test,y_train,y_test) = train_test_split(mes,cat,test_size=0.2,random_state=42)
#converting the alpha into numbers
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(x_train)#coverrtion happens for training data
#creating the model
model = MultinomialNB()
model.fit(features,y_train)


#test the model
feature_test = cv.transform(x_test)
#print(model.score(feature_test,y_test))
#predict the test
def classify(message):
    input = cv.transform([message]).toarray()
    result= model.predict(input)
    return result                           #print(result)

st.header('Spam Detection')


input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
     result = classify(input_mess)
     st.markdown(f'### Prediction: **{result[0]}**')
