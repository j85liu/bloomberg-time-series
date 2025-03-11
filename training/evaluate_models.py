import torch
import torch.nn as nn
import numpy as np
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

# Load models
def load_model(model_name, input_size):
    if model_name == "nbeatsx":
        model = NBeatsX(input_size, output_size=1)
    elif model_name == "ctts":
        model = CTTS()
    elif model_name == "tft":
        model = TemporalFusionTransformer(input_size)
    elif model_name == "deepar":
        model = DeepAR(input_size)
    else:
        raise ValueError("Invalid model name")

    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth"))
    model.to(DEVICE)
    model.eval()
    return model

# Evaluation Metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluate Models
def evaluate_models(test_data):
    X_test, Y_test = prepare_data(test_data)
    dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    models = ["nbeatsx", "ctts", "tft", "deepar"]
    results = {}

    for model_name in models:
        print(f"\nEvaluating {model_name.upper()}...\n")
        model = load_model(model_name, HPARAMS["input_size"])
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                y_pred.append(outputs.cpu().numpy().flatten())
                y_true.append(targets.numpy().flatten())

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        results[model_name] = {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "MAPE": mape(y_true, y_pred)
        }

        print(f"✅ {model_name.upper()} - RMSE: {results[model_name]['RMSE']:.4f}, "
              f"MAE: {results[model_name]['MAE']:.4f}, "
              f"MAPE: {results[model_name]['MAPE']:.2f}%")

if __name__ == "__main__":
    from data.synthetic_data import generate_multiple_series
    test_data = generate_multiple_series(num_series=1, T=252)  # Generate test data
    evaluate_models(test_data["synthetic_series_1"])
