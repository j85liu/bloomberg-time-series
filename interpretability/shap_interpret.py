import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.nbeatsx import NBeatsX
from models.ctts import CTTS
from models.tft import TemporalFusionTransformer
from models.deepar import DeepAR
from utils.preprocessing import prepare_data
from training.hyperparameters import HPARAMS

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_model(model_name):
    if model_name == "nbeatsx":
        model = NBeatsX(input_size=HPARAMS["input_size"], output_size=1)
    elif model_name == "ctts":
        model = CTTS()
    elif model_name == "tft":
        model = TemporalFusionTransformer(input_size=HPARAMS["input_size"])
    elif model_name == "deepar":
        model = DeepAR(input_size=HPARAMS["input_size"])
    else:
        raise ValueError("Invalid model name")
    
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth"))
    model.to(DEVICE)
    model.eval()
    return model

# Explain Model with SHAP
def explain_with_shap(model_name, test_data):
    print(f"\n🔍 Explaining {model_name.upper()} using SHAP...\n")

    # Prepare Data
    X_test, _ = prepare_data(test_data)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    # Load Model
    model = load_model(model_name)

    # SHAP Explainer (Gradient Explainer for Neural Networks)
    explainer = shap.GradientExplainer(model, X_test[:100])  # Use first 100 samples
    shap_values = explainer.shap_values(X_test[:10])  # Explain 10 samples

    # Visualization
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test[:10].cpu().numpy(), plot_type="bar")
    plt.show()

if __name__ == "__main__":
    from data.synthetic.synthetic_data import generate_multiple_series
    test_data = generate_multiple_series(num_series=1, T=252)  # Generate synthetic test data
    explain_with_shap("nbeatsx", test_data["synthetic_series_1"])
