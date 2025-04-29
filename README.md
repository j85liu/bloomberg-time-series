# Hybrid Time Series Forecasting for Financial Assets

This project combines multiple forecasting methods (DeepAR, Temporal Fusion Transformer, N-BEATSx) to address the weaknesses of individual models in financial asset forecasting.

## Approach

We implement several hybrid model strategies:

1. **Ensemble Methods**: Weighted combinations of individual model predictions
2. **Model Switching Framework**: Dynamically selects the best model based on market conditions
3. **Stacking/Meta-learning**: Uses a meta-model to learn optimal combinations of base forecasters
4. **Hybrid Architectures**: Combines architectural elements from different models

## File Structure

```
📂 time_series_forecasting/
├── 📂 data/                             # Data storage
│   ├── synthetic_data.py                # Generates synthetic time series data
│   └── real_data_loader.py              # Loads real financial data (e.g., from Yahoo Finance)
├── 📂 models/                           # Model implementations
│   ├── nbeatsx.py                       # N-BEATSx implementation
│   ├── ctts.py                          # CNN-Transformer model (CTTS) from JPMorgan
│   ├── tft.py                           # Temporal Fusion Transformer implementation
│   └── deepar.py                        # DeepAR implementation
├── 📂 interpretability/                 # Interpretation methods
│   ├── shap_interpret.py                # SHAP-based feature importance
│   └── lime_interpret.py                # LIME-based local interpretability
├── 📂 hybrid/                           # Hybrid model implementations
│   ├── ensemble.py                      # Ensemble methods implementation
│   ├── model_switching.py               # Model switching framework
│   ├── stacking.py                      # Stacking/meta-learning approach
│   └── hybrid_architecture.py           # Custom hybrid architecture implementation
├── 📂 training/                         # Training scripts
│   ├── train_models.py                  # Trains all models
│   └── evaluate_models.py               # Evaluates models on test data
├── 📂 utils/                            # Utility functions
│   ├── preprocessing.py                 # Data preprocessing (scaling, train-test split)
│   └── visualization.py                 # Plot predictions & interpretability results
├── main.py                              # Main script to run everything
├── requirements.txt                     # Python dependencies
└── README.md                            # Explanation of the project
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run example: `python main.py`

## Implementation Complexity

The hybrid approaches are implemented in order of increasing complexity:

1. **Ensemble Methods**: Easiest to implement, requires minimal additional code beyond the base models
2. **Model Switching**: Moderate complexity, focuses on regime detection and switching logic
3. **Stacking/Meta-learning**: Higher complexity, requires careful training of meta-models
4. **Hybrid Architectures**: Most complex, combines architectural elements for a unified model

## Hardware Requirements

- Ensemble & Model Switching: Standard CPU, minimal GPU requirements
- Stacking: Moderate GPU recommended for meta-model training
- Hybrid Architectures: Significant GPU resources required for training

## Contributions

This project aims to advance the state of financial forecasting by combining the strengths of multiple approaches while mitigating their individual weaknesses.