from flask import Flask, jsonify, request
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

# Download NLTK data if not available
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Function to preprocess text comments
def preprocess_comments(comment):
    try:
        comment = comment.lower().strip()  # Convert to lowercase and strip whitespace
        comment = re.sub(r'\n', ' ', comment)  # Remove new lines
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)  # Remove special characters

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        raise ValueError(f"Error during preprocessing: {str(e)}")

# ✅ Function to load MLflow model and vectorizer
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-13-238-120-177.ap-southeast-2.compute.amazonaws.com:5000/")
    client = MlflowClient()

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)  # Load MLflow model
    vectorizer = joblib.load(vectorizer_path)  # Load vectorizer

    return model, vectorizer

# Load the MLflow model and vectorizer
model, vectorizer = load_model_and_vectorizer('yt_chrome_plugin_model', '1', './tfidf_vectorizer.pkl')

# ✅ API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comments = data.get('comments')

    if not comments or not isinstance(comments, list):
        return jsonify({'error': 'Invalid input: Provide a list of comments'}), 400

    try:
        # Preprocess comments
        processed_comments = [preprocess_comments(comment) for comment in comments]

        # Transform using vectorizer
        transformed_comments = vectorizer.transform(processed_comments)

        # ✅ Convert sparse matrix to Pandas DataFrame (MLflow requires it)
        feature_names = vectorizer.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        # ✅ Make predictions
        predictions = model.predict(transformed_df).tolist()

    except Exception as e:
        return jsonify({"error": f"Prediction Failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

# ✅ Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
