import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data (only needs to be done once)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Define the preprocessing function
def preprocess_comment(comment):
    try:
        logger.debug(f"Preprocessing comment: {comment[:50]}...")  # Log first 50 chars

        comment = comment.lower()
        logger.debug("Lowercased comment")

        comment = comment.strip()  # Corrected typo: stripe() to strip()
        logger.debug("Removed leading/trailing whitespace")

        comment = re.sub(r'\n', '', comment)
        logger.debug("Removed newline characters")

        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)  # Keep punctuation
        logger.debug("Removed non-alphanumeric characters (except punctuation)")

        # Remove stop words but retain 'not', 'however', 'no', 'yet' for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])  # Changed to ' '.join for correct word separation
        logger.debug("Removed stop words")

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])  # Changed to ' '.join for correct word separation
        logger.debug("Lemmatized words")

        logger.debug(f"Preprocessed comment: {comment[:50]}...") # Log the preprocessed comment

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment  # Return the comment even if there's an error


def normalize_text(df):
    try:
        logger.info("Starting text normalization...")
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.info("Text normalization completed.")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise  # Re-raise the exception after logging it


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.info(f"Creating directory: {interim_data_path}")
        os.makedirs(interim_data_path, exist_ok=True)
        logger.info(f"Directory created (or already existed): {interim_data_path}")

        train_data.to_csv(os.path.join(interim_data_path, 'train_preprocessed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_preprocessed.csv'), index=False)
        logger.info(f"Preprocessed data saved to: {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def main():
    try:
        logger.info("Starting data preprocessing...")
        train_data = pd.read_csv('train.csv')  # Add your actual file paths here
        test_data = pd.read_csv('test.csv')
        logger.info("Data loaded successfully.")

        # Preprocess the data
        train_preprocessed_data = normalize_text(train_data.copy()) # Use .copy() to avoid SettingWithCopyWarning
        test_preprocessed_data = normalize_text(test_data.copy())

        # Save the preprocessed data
        save_data(train_preprocessed_data, test_preprocessed_data, data_path='chrome-plugin/data/interim')
        logger.info("Data preprocessing completed successfully.")

    except FileNotFoundError:
        logger.error("One or both input CSV files (train.csv, test.csv) not found.  Make sure they are in the correct directory.")
        print("Error: Input CSV files not found.  Please check the file paths.")
    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()