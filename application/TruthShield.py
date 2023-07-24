# imports
import streamlit as st
import joblib
import os
import string

import re

# Importing NLP Packages
import spacy
import nltk
nltk.download('wordnet')

# EDA Packages
import numpy as np
import pandas as pd

# Importing Wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Vectorizing Packages
from sklearn.feature_extraction.text import TfidfVectorizer

# For seeing and removing stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
listofstopwords = list(stopwords)
listofstopwords.extend(('said','trump','reuters','president','state','government','states','new','house','united',
                       'clinton','obama','donald','like','news','just', 'campaign', 'washington', 'election',
                        'party', 'republican', 'say','obama','(reuters)','govern','news','united', 'states', '-', 'said', 'arent', 'couldnt',
                        'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent','isnt', 'mightnt', 'mustnt', 'neednt',
                        'shant', 'shes', 'shouldnt', 'shouldve','thatll', 'wasnt', 'werent', 'wont', 'wouldnt',
                        'youd','youll', 'youre', 'youve', 'trump'))

def my_lemmatization_tokenizer(text):
    listofwords = text.split(' ')
    listoflemmatized_words = []

    for word in listofwords:
        if (not word in listofstopwords) and (word != ''):
            lemmatized_word = lemmatizer.lemmatize(word,pos='v')
            listoflemmatized_words.append(lemmatized_word)

    return listoflemmatized_words

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Vectorizing loading function
news_vectorizer = open('application/Models1/TfidfVectorizer.pkl', 'rb')
news_cv = joblib.load(news_vectorizer)

# Loading our models function
def loading_prediction_models(model_file):
    loading_prediction_models = joblib.load(open(os.path.join(model_file),'rb'))
    return loading_prediction_models

def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

def main():
    """News Classifier with Streamlit"""
    st.title('News Classifier using Machine Learning')
    st.subheader('Natural Language Processing and Machine Learning Application')
    st.markdown('**Created by Rohit**')

    activities = ['Prediction using Machine Learning', 'Natural Language Processing']

    choice = st.sidebar.selectbox("Choose Activity", activities)

    # Letting the user pick the options
    if choice == 'Prediction using Machine Learning':
        st.info('Prediction using Machine Learning')

        # Creating an input for users
        news_text = st.text_area('Enter your text below','Start typing here')
        news_text = word_drop(news_text)
        # Stating all our models
        all_ml_models = ['Logistic Regression',"Bagging CLassifier" ,'Decision Tree','Random Forest','Naive Bayes','Neural Network','AdaBoost','Support Vector Machines'] #'GridSearch Fitted']
        # Stating the model choices in a drop down
        model_choice = st.selectbox('Choose your ML Model below', all_ml_models)
        # Setting our prediction_labels dictionary for output
        prediction_labels = {'Fake News' : 0 , 'Factual News': 1}
        # Giving a button to classify text or enter command to the machine
        if st.button("Classify text"):
            # The following will output the text entered in the box (text_area) above
            st.text('Original Text ::\n{}'.format(news_text))
            # Converting the inputted text for transforming into vectors
            vect_text = news_cv.transform([news_text]).toarray()

            # If user selects Logistic Regression
            if model_choice == 'Logistic Regression':
                # Importing the model to predict
                predictor = loading_prediction_models('Models1/LR_model.pkl')
                # Setting our prediction by calling .predict on the model selected
                prediction = predictor.predict(vect_text)

            # If user chooses Decision Tree Classifier
            elif model_choice == "Bagging CLassifier":
                predictor = loading_prediction_models('Models1/Bag_model.pkl')
                prediction = predictor.predict(vect_text)

            elif model_choice == 'Decision Tree':
                predictor = loading_prediction_models('Models1/DT_model.pkl')
                prediction = predictor.predict(vect_text)

            # If user chooses Random Forest Classifier
            elif model_choice == 'Random Forest':
                predictor = loading_prediction_models('Models1/RF_model.pkl')
                prediction = predictor.predict(vect_text)

            elif model_choice == 'Naive Bayes':
                predictor = loading_prediction_models('Models1/NB_model.pkl')
                prediction = predictor.predict(vect_text)

            elif model_choice == 'Neural Network':
                predictor = loading_prediction_models('Models1/NN_model.pkl')
                prediction = predictor.predict(vect_text)

            elif model_choice == 'Support Vector Machines':
                predictor = loading_prediction_models('Models1/SVC_model.pkl')
                prediction = predictor.predict(vect_text)
            elif model_choice == "AdaBoost":
                predictor = loading_prediction_models('Models1/ADB_model.pkl')
                prediction = predictor.predict(vect_text)

            final_result = get_keys(prediction[0], prediction_labels)
            st.success('News Categorized as:: {}'.format(final_result))

    # If the user decides to choose NLP
    if choice == 'Natural Language Processing':
        st.info('Natural Language Processing')
        news_text = st.text_area('Enter your text below','Start typing here')
        news_text = word_drop(news_text)
        nlp_task = ['Tokenization', 'Lemmatization']
        task_choice = st.selectbox('Choose NLP task', nlp_task)
        if st.button('Analyze'):
            st.info('Original Text ::\n {}'.format(news_text))

            docx = my_lemmatization_tokenizer(news_text) if task_choice == 'Lemmatization' else news_text.split(' ')

            st.json({'Tokens': docx})

        # Giving a button to put it in a tabular format
        if st.button("Tabulize"):
            docx = my_lemmatization_tokenizer(news_text) if task_choice == 'Lemmatization' else news_text.split(' ')
            c_tokens = docx
            c_lemma = [lemmatizer.lemmatize(token) for token in docx]

            new_df = pd.DataFrame(zip(c_tokens, c_lemma), columns=['Tokens', 'Lemmatized Words'])
            st.dataframe(new_df)

        if st.checkbox('Wordcloud'):
            wordcloud = WordCloud().generate(news_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot()

# This will be the last line
if __name__ == '__main__':
    main()
