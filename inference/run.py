"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder



# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json" 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

class IrisNet(nn.Module):

    def __init__(self) -> None:
        super(IrisNet, self).__init__()

        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str) -> IrisNet:
    """Loads and returns the specified model"""
    model = IrisNet()
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        logging.info(f'Path of the model {path}')
        return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model, infer_data_loader: DataLoader) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""
    predictions = []
    with torch.no_grad():
        for batch in infer_data_loader:
            inputs = batch[0]
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
    
    with open('inference/inference_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    class_names = encoder.inverse_transform(predictions)

    return class_names

def process_data(infer_data: pd.DataFrame):

    scaled_df, target = scale_label(infer_data)

    X_tensor, y_tensor = split_tensor(scaled_df, target)

    dataset_loader = load_dataset(X_tensor, y_tensor)

    return dataset_loader


def scale_label(df):
    logging.info("Scaling and Labeling Data...")

    features_scaled = df.drop(columns='Species')
    scaled_df = StandardScaler().fit_transform(features_scaled)

    encoder = LabelEncoder()
    target = encoder.fit_transform(df['Species'])

    # Save the fitted LabelBinarizer for later use
    save_label_encoder(encoder)

    return scaled_df, target

def save_label_encoder(encoder):
    with open('inference/inference_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

def split_tensor( X, y):    
    logging.info("Converting Sets into Tensors...")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

def load_dataset(X_tensor, y_tensor,):
    logging.info("Loading Data into Dataloader...")
    dataset = TensorDataset(X_tensor, y_tensor)
    dataset_loader = DataLoader(dataset, batch_size=conf['train']['batch_size'])
    return dataset_loader


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    infer_data_loader = process_data(infer_data)
    results = predict_results(model, infer_data_loader)

    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()