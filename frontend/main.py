import streamlit as st
import pickle
import nltk
import string

model = pickle.load(open('Model/model.pkl', 'rb'))
tfidf = pickle.load(open('Model/vectorizer.pkl', 'rb'))
porterstemmer = pickle.load(open('Model/stemmer.pkl', 'rb'))


def preprocess_text(text):
    text = text.lower()
    tokenized_data = nltk.word_tokenize(text)
    y = []
    for i in tokenized_data:
        if i.isalnum() and i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            y.append(porterstemmer.stem(word=i))
    text = y[:]
    return " ".join(text)


st.title("SMS classifier")
data = st.text_area("Enter the SMS")
if st.button('Predict'):
    preprocessed_data = preprocess_text(data)
    vectorized_input = tfidf.transform([preprocessed_data])
    prediction = model.predict(vectorized_input)[0]
    if prediction:
        st.header("Spam")
    else:
        st.header("Not Spam")
