import lime
import lime.lime_tabular
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

# LIME Explainer
def explain_with_lime(model_name, test_data):
    print(f"\n🔍 Explaining {model_name.upper()} using LIME...\n")

    # Prepare Data
    X_test, _ = prepare_data(test_data)
    X_test = X_test[:10]  # Select a few samples

    # Load Model
    model = load_model(model_name)

    # Define Prediction Function
    def predict_fn(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            return model(X_tensor).cpu().numpy()

    # LIME Tabular Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test, mode="regression", feature_names=[f"Day {i}" for i in range(X_test.shape[1])]
    )

    # Explain First Instance
    instance = X_test[0]
    exp = explainer.explain_instance(instance, predict_fn, num_features=5)
    
    # Show Explanation
    exp.show_in_notebook()
    exp.as_pyplot_figure()
    plt.show()

if __name__ == "__main__":
    from data.synthetic_data import generate_multiple_series
    test_data = generate_multiple_series(num_series=1, T=252)  # Generate synthetic test data
    explain_with_lime("nbeatsx", test_data["synthetic_series_1"])
