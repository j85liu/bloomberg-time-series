File Structure

📂 time_series_forecasting  
│── 📂 data/                      # Data storage  
│   │── synthetic_data.py         # Generates synthetic time series data  
│   │── real_data_loader.py       # Loads real financial data (e.g., from Yahoo Finance)  
│── 📂 models/                    # Model implementations  
│   │── nbeatsx.py                # N-BEATSx implementation  
│   │── ctts.py                   # CNN-Transformer model (CTTS) from JPMorgan  
│   │── tft.py                    # Temporal Fusion Transformer implementation  
│   │── deepar.py                  # DeepAR implementation  
│── 📂 interpretability/          # Interpretation methods  
│   │── shap_interpret.py         # SHAP-based feature importance  
│   │── lime_interpret.py         # LIME-based local interpretability  
│── 📂 training/                   # Training scripts  
│   │── train_models.py           # Trains all models  
│   │── evaluate_models.py        # Evaluates models on test data  
│── 📂 utils/                      # Utility functions  
│   │── preprocessing.py          # Data preprocessing (scaling, train-test split)  
│   │── visualization.py          # Plot predictions & interpretability results  
│── main.py                       # Main script to run everything  
│── requirements.txt              # Python dependencies  
│── README.md                     # Explanation of the project  
