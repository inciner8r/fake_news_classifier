 
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Download stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('kaggle_fake_train.csv')

# Cleaning the news
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

# Creating the Bag of Words model
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

# Extracting dependent variable from the dataset
y = df['label']

# Training the Logistic Regression model
classifier = LogisticRegression(C=0.8, random_state=0)
classifier.fit(X, y)

# Save the trained model as a pickle file
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Save the CountVectorizer object as a pickle file
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)
