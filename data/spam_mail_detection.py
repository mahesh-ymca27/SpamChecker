import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utils import process_text # ✅ Import the function from utils.py

# Download stopwords once
nltk.download('stopwords')

# Load the data
df = pd.read_csv('spam_ham_dataset.csv')

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check for missing data
df.isnull().sum()

# Initialize and fit the CountVectorizer
cv = CountVectorizer(analyzer=process_text)
messages_bow = cv.fit_transform(df['text'])

# Save the fitted vectorizer
joblib.dump(cv, 'vectorizer.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['label'], test_size=0.20, random_state=0)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predictions and Evaluation on the training set
train_pred = classifier.predict(X_train)
print('Training Set Evaluation:')
print(classification_report(y_train, train_pred))
print('Confusion Matrix: \n', confusion_matrix(y_train, train_pred))
print('Accuracy: ', accuracy_score(y_train, train_pred))

# Predictions and Evaluation on the test set
test_pred = classifier.predict(X_test)
print('Test Set Evaluation:')
print(classification_report(y_test, test_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, test_pred))
print('Accuracy: ', accuracy_score(y_test, test_pred))

# Save the trained model
joblib.dump(classifier, 'finalized_model.sav')
