from interpretability.shap_explainer import explain_with_shap
from interpretability.lime_explainer import explain_with_lime
from data.synthetic_data import generate_multiple_series

# Generate Synthetic Test Data
test_data = generate_multiple_series(num_series=1, T=252)
test_series = test_data["synthetic_series_1"]

# Run SHAP and LIME for all models
models = ["nbeatsx", "ctts", "tft", "deepar"]

for model in models:
    explain_with_shap(model, test_series)
    explain_with_lime(model, test_series)
