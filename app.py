import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt

# Download stopwords
nltk.download('stopwords')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('kaggle_fake_train.csv')
    return df

# Cleaning the news
@st.cache_data
def preprocess_data(df):
    corpus = []
    ps = PorterStemmer()

    for i in range(df.shape[0]):
        # Check if the title is NaN
        if pd.isna(df.title[i]):
            title = ""  # If NaN, assign an empty string
        else:
            # Cleaning special character from the news-title
            title = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.title[i])
            # Converting the entire news-title to lower case
            title = title.lower()
            # Tokenizing the news-title by words
            words = title.split()
            # Removing the stopwords
            words = [word for word in words if word not in set(stopwords.words('english'))]
            # Stemming the words
            words = [ps.stem(word) for word in words]
            # Joining the stemmed words
            title = ' '.join(words)

        # Building a corpus of news-title
        corpus.append(title)

    return corpus

# Creating the Bag of Words model
@st.cache_data
def create_bag_of_words(corpus):
    cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
    X = cv.fit_transform(corpus).toarray()
    return cv, X

# Training the Logistic Regression model
@st.cache_data
def train_logistic_regression_model(X_train, y_train):
    classifier = LogisticRegression(C=0.8, random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

# Training the Multinomial Naive Bayes model
@st.cache_data
def train_multinomial_nb_model(X_train, y_train):
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    return classifier

# Calculate evaluation metrics
@st.cache_data
def calculate_metrics(_classifier, X_test, y_test):
    y_pred = _classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

# Define function to display evaluation metrics
def display_metrics(accuracy, precision, recall, f1, cm, model_name):
    st.subheader(f"{model_name} Model Evaluation")
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)

    # Displaying confusion matrix
    st.subheader("Confusion Matrix")
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Streamlit app
def main():
    df = load_data()
    corpus = preprocess_data(df)
    cv, X = create_bag_of_words(corpus)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logistic_regression_classifier = train_logistic_regression_model(X_train, y_train)
    multinomial_nb_classifier = train_multinomial_nb_model(X_train, y_train)

    st.title("Fake News Detection")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Prediction App", "Model Evaluation"))

    if page == "Prediction App":
        display_prediction_app(logistic_regression_classifier, multinomial_nb_classifier, cv)
    elif page == "Model Evaluation":
        display_metrics_for_both_models(logistic_regression_classifier, multinomial_nb_classifier, X_test, y_test)

# Define function to display prediction app
def display_prediction_app(logistic_regression_classifier, multinomial_nb_classifier, cv):
    st.title("Fake News Detection")
    st.write("Enter a news headline below to predict whether it's fake or not:")
    news_headline = st.text_input("Enter News Headline")
    if st.button("Predict"):
        if news_headline.strip() == "":
            st.warning("Please enter a news headline.")
        else:
            logistic_regression_prediction = predict_fake_news(news_headline, logistic_regression_classifier, cv)
            multinomial_nb_prediction = predict_fake_news(news_headline, multinomial_nb_classifier, cv)

            st.subheader("Logistic Regression Model Prediction:")
            if logistic_regression_prediction == 1:
                st.error("Fake News Detected!")
            else:
                st.success("No Fake News Detected.")

            st.subheader("Multinomial Naive Bayes Model Prediction:")
            if multinomial_nb_prediction == 1:
                st.error("Fake News Detected!")
            else:
                st.success("No Fake News Detected.")

# Predict function
def predict_fake_news(news_headline, classifier, cv):
    ps = PorterStemmer()
    title = re.sub(pattern='[^a-zA-Z]', repl=' ', string=news_headline)
    title = title.lower()
    words = title.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    title = ' '.join(words)
    temp = cv.transform([title]).toarray()
    return classifier.predict(temp)[0]

# Define function to display evaluation metrics for both models
def display_metrics_for_both_models(logistic_regression_classifier, multinomial_nb_classifier, X_test, y_test):
    logistic_regression_accuracy, logistic_regression_precision, logistic_regression_recall, logistic_regression_f1, logistic_regression_cm = calculate_metrics(logistic_regression_classifier, X_test, y_test)
    display_metrics(logistic_regression_accuracy, logistic_regression_precision, logistic_regression_recall, logistic_regression_f1, logistic_regression_cm, "Logistic Regression")

    multinomial_nb_accuracy, multinomial_nb_precision, multinomial_nb_recall, multinomial_nb_f1, multinomial_nb_cm = calculate_metrics(multinomial_nb_classifier, X_test, y_test)
    display_metrics(multinomial_nb_accuracy, multinomial_nb_precision, multinomial_nb_recall, multinomial_nb_f1, multinomial_nb_cm, "Multinomial Naive Bayes")

if __name__ == "__main__":
    main()
