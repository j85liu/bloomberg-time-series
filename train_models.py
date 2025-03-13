import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.nbeatsx import NBeatsX
from models.ctts import CTTS
from models.tft import TemporalFusionTransformer
from models.deepar import DeepAR
from utils.preprocessing import prepare_data
from training.hyperparameters import HPARAMS

# ✅ Set working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

# ✅ Set device (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define data directory
DATA_DIR = "data/synthetic"
SAVE_DIR = "saved_models"

# 🔹 Training function
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    """
    Trains a given model on time-series data.
    """
    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

    return model

# 🔹 Function to train models on all datasets
def train_all_models():
    """
    Loads existing synthetic datasets and trains N-BEATSx, CTTS, TFT, and DeepAR models.
    """
    if not os.path.exists(DATA_DIR):
        print("❌ No synthetic data found. Please generate it first.")
        return
    
    # ✅ Find all available synthetic data files
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    
    if not csv_files:
        print("❌ No synthetic data files found. Please generate data.")
        return
    
    print(f"✅ Found {len(csv_files)} synthetic datasets. Starting training...")

    # ✅ Ensure save directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for file in csv_files:
        series_name = file.replace(".csv", "")
        file_path = os.path.join(DATA_DIR, file)
        print(f"\n📌 Processing {series_name} ({file_path})...\n")

        # ✅ Load dataset
        df = pd.read_csv(file_path)
        X_train, Y_train = prepare_data(df, HPARAMS["input_size"])
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
        train_loader = DataLoader(dataset, batch_size=HPARAMS["batch_size"], shuffle=True)

        # ✅ Define Models
        models = {
            "nbeatsx": NBeatsX(input_size=HPARAMS["input_size"], output_size=1).to(DEVICE),
            "ctts": CTTS().to(DEVICE),
            "tft": TemporalFusionTransformer(input_size=HPARAMS["input_size"]).to(DEVICE),
            "deepar": DeepAR(input_size=HPARAMS["input_size"]).to(DEVICE),
        }

        criterion = nn.MSELoss()

        # 🔹 Train and save each model
        for name, model in models.items():
            print(f"\n🚀 Training {name.upper()} on {series_name}...\n")
            optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])
            trained_model = train_model(model, train_loader, criterion, optimizer, HPARAMS["epochs"])

            # ✅ Save model
            model_save_path = os.path.join(SAVE_DIR, f"{name}_{series_name}.pth")
            torch.save(trained_model.state_dict(), model_save_path)
            print(f"✅ {name.upper()} trained on {series_name} saved at {model_save_path}.")

if __name__ == "__main__":
    train_all_models()
