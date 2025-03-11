import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import DataLoader, TensorDataset
from models.nbeatsx import NBeatsX
from models.ctts import CTTS
from models.tft import TemporalFusionTransformer
from models.deepar import DeepAR
from utils.preprocessing import prepare_data
from training.hyperparameters import HPARAMS

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save best hyperparameters
BEST_HPARAMS_PATH = "tuning/optimized_hparams.json"

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """
    Trains model for given hyperparameters and returns validation loss.
    """
    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def objective(trial):
    """
    Optuna objective function to optimize hyperparameters.
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_units = trial.suggest_int("hidden_units", 32, 512)

    # Prepare Data (Split into train & validation)
    from data.synthetic_data import generate_multiple_series
    data = generate_multiple_series(num_series=1, T=252)
    train_data, val_data = data["synthetic_series_1"][:200], data["synthetic_series_1"][200:]

    X_train, Y_train = prepare_data(train_data)
    X_val, Y_val = prepare_data(val_data)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Choose Model
    model_name = trial.suggest_categorical("model", ["nbeatsx", "ctts", "tft", "deepar"])
    if model_name == "nbeatsx":
        model = NBeatsX(input_size=HPARAMS["input_size"], output_size=1, hidden_units=hidden_units)
    elif model_name == "ctts":
        model = CTTS(hidden_units=hidden_units)
    elif model_name == "tft":
        model = TemporalFusionTransformer(input_size=HPARAMS["input_size"], hidden_units=hidden_units)
    elif model_name == "deepar":
        model = DeepAR(input_size=HPARAMS["input_size"], hidden_units=hidden_units)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train & Validate
    val_loss = train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    return val_loss

# Run Optuna
def run_hyperparameter_tuning(n_trials=20):
    """
    Runs Optuna hyperparameter tuning and saves best parameters.
    """
    print("\n🔍 Running Hyperparameter Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Save best hyperparameters
    best_hparams = study.best_params
    with open(BEST_HPARAMS_PATH, "w") as f:
        json.dump(best_hparams, f, indent=4)

    print("\n✅ Best Hyperparameters Found & Saved!")
    print(best_hparams)

if __name__ == "__main__":
    run_hyperparameter_tuning()
