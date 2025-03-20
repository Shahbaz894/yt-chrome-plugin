import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the trained TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return classification metrics and confusion matrix."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed')
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def main():
    """Main function to execute model evaluation and log results to MLflow."""
    
    
    mlflow.set_tracking_uri("http://ec2-13-238-120-177.ap-southeast-2.compute.amazonaws.com:5000/")
   


    mlflow.set_experiment('dvc-pipeline-runs-project')

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.getcwd())  # Get root directory

            # Load parameters from params.yaml
            params_path = os.path.join(root_dir, "params.yaml")
            params = load_params(params_path)

            # Log parameters in MLflow
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            logger.debug('lgbm_model.pkl model loaded successfully')

            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            logger.debug('tfidf_vectorizer.pkl loaded successfully')

            # Load test data
            test_data = load_data(os.path.join(root_dir, 'chrome-plugin/data/interim/test_preprocessed.csv'))
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Infer the model signature
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))

            # Log model to MLflow
            mlflow.sklearn.log_model(model, 'lgbm_model', signature=signature, input_example=input_example)

            # Save model info
            save_model_info(run.info.run_id, 'lgbm_model', 'experiment_info.json')

            # Log TF-IDF vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            # Evaluate the model
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification metrics properly
            for label, metrics in report.items():
                if isinstance(metrics, dict):  # Ensure it is a dictionary
                    mlflow.log_metric(f"test_{label}_precision", metrics['precision'])
                    mlflow.log_metric(f"test_{label}_recall", metrics['recall'])
                    mlflow.log_metric(f"test_{label}_f1-score", metrics['f1-score'])

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
