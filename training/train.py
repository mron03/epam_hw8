"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Comment this lines if you have problems with MLFlow installation
import mlflow
mlflow.autolog()

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
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)

        X, y = self.scale_label(df)

        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = self.split_tensor(X, y)

        train_loader, test_loader = self.dataset_loader(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

        return train_loader, test_loader

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def scale_label(self, df):
        logging.info("Scaling and Labeling Data...")

        features_scaled = df.drop(columns='Species')
        scaled_df = StandardScaler().fit_transform(features_scaled)

        self.encoder = LabelEncoder()
        target = self.encoder.fit_transform(df['Species'])

        # Save the fitted LabelBinarizer for later use
        self.save_label_encoder()

        return scaled_df, target
    
    def save_label_encoder(self):
        with open('training/train_encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder, f)

    def split_tensor(self, X, y):
        logging.info("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42)
        
        logging.info("Converting Sets into Tensors...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def dataset_loader(self, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor):
        logging.info("Loading Data into Dataloader...")

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=conf['train']['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=conf['train']['batch_size'])

        return train_loader, test_loader


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
    


class Training():

    def __init__(self) -> None:
        self.model = IrisNet()

    def run_training(self, train_loader: DataLoader, test_loader: DataLoader, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        start_time = time.time()

        self.train(train_loader)

        end_time = time.time()

        logging.info(f"Training completed in {end_time - start_time} seconds.")

        self.test(test_loader)
        self.save(out_path)
    
    def train(self, train_loader) -> None:
        logging.info("Training the model...")
        
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        num_epochs = conf['train']['num_epochs']
        for epoch in range(num_epochs):

            for inp, labels in train_loader:
                optimizer.zero_grad()

                outputs = self.model.forward(inp)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


    def test(self, test_loader: DataLoader) -> float:
        logging.info("Testing the model...")
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inp, labels in test_loader:
                outputs = self.model(inp)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        with open('training/train_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        class_report = classification_report(all_labels, all_preds, target_names=encoder.classes_)

        logging.info(f"Classification Report: {class_report}")
        return class_report

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(self.model.state_dict(), path)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    train_loader, test_loader = data_proc.prepare_data()
    tr.run_training(train_loader, test_loader)


if __name__ == "__main__":
    main()