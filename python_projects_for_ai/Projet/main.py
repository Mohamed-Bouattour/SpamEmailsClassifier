import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv("C:/Users/DELL/Downloads/spam.csv")
data["spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)
x_train, x_test, y_train, y_test = train_test_split(data.Message, data.spam)

# Vectorize data
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

# Train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# title of streamlit applicatioon
st.title('Spam Detection App')

# Text input for user input
user_input = st.text_input('Enter an email message:')

# Button for making prediction
if st.button('Predict'):
    # Vectorize user input
    user_input_count = cv.transform([user_input])
    # Make prediction
    prediction = model.predict(user_input_count)
    # Display prediction
    if prediction[0] == 1:
        st.write('This is a spam email.')
    else:
        st.write('This is not a spam email.')
