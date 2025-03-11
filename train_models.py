import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from models.nbeatsx import NBeatsX
from models.ctts import CTTS
from models.tft import TemporalFusionTransformer
from models.deepar import DeepAR
from utils.preprocessing import prepare_data  # Data preparation function
from training.hyperparameters import HPARAMS  # Load hyperparameters

# 📌 Set script's directory as working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)  # Change working directory
sys.path.append(PROJECT_ROOT)  # Ensure project modules can be imported

# ✅ Set device (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_all_models(train_data, save_path="saved_models/"):
    """
    Trains N-BEATSx, CTTS, TFT, and DeepAR models.
    """
    # 📌 Ensure correct save directory
    save_path = os.path.join(PROJECT_ROOT, save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Prepare Data
    X_train, Y_train = prepare_data(train_data)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=HPARAMS["batch_size"], shuffle=True)

    # Define Models
    models = {
        "nbeatsx": NBeatsX(input_size=HPARAMS["input_size"], output_size=1).to(DEVICE),
        "ctts": CTTS().to(DEVICE),
        "tft": TemporalFusionTransformer(input_size=HPARAMS["input_size"]).to(DEVICE),
        "deepar": DeepAR(input_size=HPARAMS["input_size"]).to(DEVICE),
    }

    # Loss Function & Optimizer
    criterion = nn.MSELoss()

    for name, model in models.items():
        print(f"\n🚀 Training {name.upper()}...\n")
        optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])
        trained_model = train_model(model, train_loader, criterion, optimizer, HPARAMS["epochs"])
        
        # Save model
        model_save_path = os.path.join(save_path, f"{name}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"✅ Model {name} saved at {model_save_path}.")

if __name__ == "__main__":
    from data.synthetic_data import generate_multiple_series
    synthetic_data = generate_multiple_series(num_series=1, T=252)  # Generate 1 year of data
    train_all_models(synthetic_data["synthetic_series_1"])
