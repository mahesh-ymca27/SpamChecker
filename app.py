from flask import Flask, render_template, url_for, request
import pandas as pd 
from nltk.corpus import stopwords
import string 
import joblib
import nltk
nltk.download('stopwords')  # Download once at the start


app = Flask(__name__)
# Tokenization function
def process_text(text):
    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove stop words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    # Return clean words
    return clean_words

# Load the trained model and vectorizer once at the start of the app
classifier = joblib.load('finalized_model.sav')  # Load the saved Naive Bayes model
vectorizer = joblib.load('vectorizer.pkl')  # Load the saved CountVectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # When the user submits a comment, it will be transformed using the vectorizer and predicted using the classifier
    if request.method == 'POST':
        comment = request.form['comment']  # Get the comment from the form
        
        # Check if the comment is empty
        if not comment.strip():  # If the comment is empty or only contains spaces
            return render_template('results.html', prediction="empty")  # Return empty message

        data = [comment]  # Convert to a list (the vectorizer expects a list)
        
        # Transform the input using the loaded vectorizer
        vect = vectorizer.transform(data).toarray()  # Use the pre-fitted vectorizer to transform the data
        
        # Predict the class (spam/ham) using the loaded classifier
        my_prediction = classifier.predict(vect)

        prediction_result = my_prediction[0]  # No conversion to int

        # Render the results in the results.html template
        return render_template('results.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
