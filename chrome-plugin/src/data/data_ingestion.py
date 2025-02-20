import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
import os
from pathlib import Path


# Setup logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        df.dropna(inplace=True)  # Remove missing values
        df.drop_duplicates(inplace=True)  # Remove duplicates
        df = df[df['clean_comment'].str.strip() != '']  # Remove empty strings

        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


    """Save the train and test data, ensuring the directory exists."""
    try:
        # Convert data_path to an absolute path
        base_path = Path(data_path).resolve()
        raw_data_path = base_path / "raw"

        # Ensure raw directory exists
        raw_data_path.mkdir(parents=True, exist_ok=True)

        # Define file paths
        train_file = raw_data_path / "train.csv"
        test_file = raw_data_path / "test.csv"

        # Print paths for debugging
        print(f"📂 Data will be saved in: {raw_data_path}")
        print(f"📄 Train file path: {train_file}")
        print(f"📄 Test file path: {test_file}")

        # Save data
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        # Confirm if files exist after saving
        if train_file.exists() and test_file.exists():
            print("✅ Train and test data successfully saved!")
        else:
            print("❌ Error: Files were not saved correctly!")

    except PermissionError:
        print("❌ PermissionError: Check if you have write access to the directory.")
    except FileNotFoundError:
        print("❌ FileNotFoundError: The specified directory does not exist.")
    except Exception as e:
        print(f"❌ Unexpected error while saving data: {e}")




def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Saves train and test data as CSV files in the specified data path with logging at each step.
    
    :param train_data: Pandas DataFrame containing training data
    :param test_data: Pandas DataFrame containing testing data
    :param data_path: Path to save the data files
    """
    try:
        logger.info(f"Checking if directory exists: {data_path}")
        
        # Ensure the directory exists
        if not os.path.exists(data_path):
            logger.info(f"Directory does not exist. Creating: {data_path}")
            os.makedirs(data_path, exist_ok=True)

        # Define file paths
        train_file = os.path.join(data_path, "train.csv")
        test_file = os.path.join(data_path, "test.csv")

        # Save train data
        logger.info(f"Saving train data to {train_file}")
        train_data.to_csv(train_file, index=False)
        logger.info(f"Train data saved successfully: {train_file}")

        # Save test data
        logger.info(f"Saving test data to {test_file}")
        test_data.to_csv(test_file, index=False)
        logger.info(f"Test data saved successfully: {test_file}")

        # Verify files are written
        saved_files = os.listdir(data_path)
        logger.info(f"Files in '{data_path}': {saved_files}")

        if "train.csv" not in saved_files or "test.csv" not in saved_files:
            logger.warning("CSV files not found after saving. Please check permissions or refresh the folder.")

    except Exception as e:
        logger.error(f"Error saving data: {e}", exc_info=True)

    """
    Saves train and test data as CSV files in the specified data path with logging at each step.
    
    :param train_data: Pandas DataFrame containing training data
    :param test_data: Pandas DataFrame containing testing data
    :param data_path: Path to save the data files
    """

    try:
        logging.info(f"Checking if directory exists: {data_path}")
        if not os.path.exists(data_path):
            logging.info(f"Directory does not exist. Creating: {data_path}")
            os.makedirs(data_path, exist_ok=True)

        # Define file paths
        train_file = os.path.join(data_path, "train.csv")
        test_file = os.path.join(data_path, "test.csv")

        # Save train data
        logging.info(f"Saving train data to {train_file}")
        train_data.to_csv(train_file, index=False)
        logging.info(f"Train data saved successfully: {train_file}")

        # Save test data
        logging.info(f"Saving test data to {test_file}")
        test_data.to_csv(test_file, index=False)
        logging.info(f"Test data saved successfully: {test_file}")

        # Verify that files are written
        saved_files = os.listdir(data_path)
        logging.info(f"Files in '{data_path}': {saved_files}")

        if "train.csv" not in saved_files or "test.csv" not in saved_files:
            logging.warning("CSV files not found after saving. Please check permissions or refresh the folder.")

    except Exception as e:
        logging.error(f"Error saving data: {e}", exc_info=True)

def main():
    try:
        # Get project root directory
        PROJECT_ROOT = os.path.abspath(os.getcwd())  # Safer than using __file__

        # Define the correct path to params.yaml
        params_path = os.path.join(PROJECT_ROOT, "params.yaml")

        # Load parameters from params.yaml
        params = load_params(params_path)

        # Access `test_size`
        test_size = params['data_ingestion']['test_size']

        # Load data from the specified URL
        df = load_data(data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/main/data/reddit.csv')

        # Preprocess the data
        final_df = preprocess_data(df)

        # Split the data
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the split datasets
        save_data(train_data, test_data, data_path=os.path.join(PROJECT_ROOT, "chrome-plugin/data/raw"))

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

