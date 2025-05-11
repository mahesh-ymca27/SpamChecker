
## 📄 SpamChecker - Spam Message Detection Web App

**SpamChecker** is a simple yet effective web application built using **Flask**, designed to classify SMS or email messages as **Spam** or **Not Spam** using a trained **Naïve Bayes** machine learning model. The app allows users to input a message and instantly get the prediction result through a clean, browser-based interface.

---

### 🚀 Features

* ✅ Spam detection using a trained Naïve Bayes classifier
* ✅ Integrated Flask backend for prediction logic
* ✅ NLTK-based text preprocessing (punctuation removal, stopword filtering)
* ✅ Pre-trained model and vectorizer loaded using Joblib
* ✅ User-friendly web interface for real-time predictions

---

### 🧠 Machine Learning Workflow

* **Text Preprocessing**: Removes punctuation and stopwords using NLTK
* **Vectorization**: CountVectorizer used to convert text into numerical features
* **Model**: Trained Naïve Bayes classifier (`finalized_model.sav`)
* **Evaluation Metrics**: High accuracy, precision, and F1-score (e.g., 97.3% accuracy)

---

### 🗂️ Project Structure

```
SpamChecker/
├── app.py                      # Flask backend
├── finalized_model.sav         # Trained ML model
├── vectorizer.pkl              # CountVectorizer object
├── templates/
│   ├── index.html              # Input form
│   └── results.html            # Prediction result
├── static/                     # CSS/JS (optional)
└── Project Report.pdf          # Report document (optional)
```

---

### 🛠️ Installation & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/SpamChecker.git
   cd SpamChecker
   ```

---
Here's a detailed section you can add to your README under the heading **"## 🚀 Embedding Flask into `app.py`"**:

---

## 🚀 Embedding Flask into `app.py`

The `app.py` file is the core of the **SpamChecker** project. It integrates the **Flask** framework to serve a lightweight and efficient web interface, enabling users to interact with the trained spam classifier model in real time. Here's how the integration works:

### 2. Key Components in `app.py`:

1. **Importing Required Libraries**

   ```python
   from flask import Flask, render_template, request
   import pandas as pd
   import string
   import joblib
   from nltk.corpus import stopwords
   ```

2. **Initialize Flask App**

   ```python
   app = Flask(__name__)
   ```

3. **Text Preprocessing Function**

   * A custom function `process_text()` is defined to:

     * Remove punctuation
     * Remove English stopwords using NLTK
   * Prepares input text for vectorization

   ```python
   def process_text(text):
       ...
   ```

4. **Loading Pre-trained Model & Vectorizer**

   * Naïve Bayes model and CountVectorizer are loaded using `joblib`

   ```python
   classifier = joblib.load('finalized_model.sav')
   vectorizer = joblib.load('vectorizer.pkl')
   ```

5. **Route: Home Page (`/`)**

   * Renders the main page `index.html` where the user inputs text

   ```python
   @app.route('/')
   def home():
       return render_template('index.html')
   ```

6. **Route: Prediction (`/predict`)**

   * Captures user input from the form
   * Transforms it using the preloaded vectorizer
   * Predicts using the Naïve Bayes model
   * Returns result via `results.html`

   ```python
   @app.route('/predict', methods=['POST'])
   def predict():
       ...
   ```

7. **Running the Flask App**

   ```python
   if __name__ == '__main__':
       app.run(debug=True)
   ```

---

📌 **Note**: The Flask server runs locally by default at `http://127.0.0.1:5000/`. You can modify the host or port if deploying.


#### 3: Implement Index.html

  

    <div class="get-start-area">
        <!-- Form Start -->
    		    <form action="{{ url_for('predict')}}" method="POST" class="form-inline">
    		    <textarea name="comment" class="form-control" id="comment" cols="50" rows="4"
		    	 placeholder="Enter Text Message*" required></textarea>
    		    <input type="submit" class="submit" value="Check Spam or Not!">
    		    </form>
        <!-- Form End --> 
    </div>

The key to note is the action attribute. It is set to **"{{ url_for('predict')}}".** So, when a user enters an E-mail and submits it. The POST method stores it in the **name** attribute named "comment" in the above html code & then passes it to the render_template() function.

#### Step 7: Implement Results.html

{% if prediction == 'spam' %}
    <h2 style="color: white;">Spam</h2>
{% elif prediction == 'ham' %}
    <h2 style="color: white;">Not a Spam!</h2>
{% endif %}


The prediction containers will display the predicted spam or not spam result. These values will be passed in via the render_template() function in the app.py file.

#### Step 8: Run the Application to Localhost
Now, go to the project directory in the CMD and type:

    python app.py

This model will running on a web browser on the localhost. The last task is to deploy it on an external server so that the public can access it.(Optional) 

## Project Members

**Mahesh Kadawla**

*B.Tech | Computer Engineering | J.C Bose University of Science and Technology YMCA Fbd.
mhesh.021@gmail.com*

