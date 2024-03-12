# to run use (" python -m streamlit run app.py ")
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading the saved vectorizer and Naive Bayes model 
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
#transform_text function for text preprocessing 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower() #converting to lowercase
    text = nltk.word_tokenize(text)  #Tokenize
    
    #Removing special charcters and retianing alpanumeric words
    text = [word for word in text if word.isalnum()]
    #Removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Applying stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

#saving stream lit code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")


if st.button('Predict'):
    #preprocess
    transformed_sms = transform_text(input_sms)
    #vectorize
    vector_input = tfidf.transform([transformed_sms])
    #predict
    result = model.predict(vector_input)[0]
    #display

    if result == 1:
        st.header("spam")
    else:
        st.header("Spam")
    
