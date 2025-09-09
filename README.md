# Viscosity-Prediction-for-Deep-Eutectic-Solvents
This repository contains the complete workflow for viscosity prediction in deep eutectic solvents (DESs), integrating systematic data preprocessing, molecular descriptor generation (Mordred–RDKit), and advanced machine learning models. The pipeline spans dataset curation, feature engineering, model training (MLP, Random Forest, XGBoost, and Evidential Deep Learning), and evaluation with cross-validation and uncertainty quantification. It provides a reproducible framework for developing accurate and interpretable property prediction models in complex solvent systems.
✨ Overview
This repository presents a comprehensive machine learning approach for predicting the viscosity of Deep Eutectic Solvents (DES) using molecular descriptors and advanced neural network architectures.

Dataset: 2,229 experimental viscosity records for 183 DES systems (from Yu et al., 2022)
Descriptors: 3,761 molecular features generated using Mordred–RDKit from SMILES retrieved via PubChem PUG REST and CACTUS services
Final Feature Set: 102 curated descriptors + auxiliary fields after cleaning, outlier removal, and feature selection
Models:

MLP: Nonlinear regression with dropout, weight decay, and Optuna-based hyperparameter tuning
EDL: Evidential Deep Learning for joint viscosity prediction and uncertainty quantification (Normal–Inverse–Gamma output)
Baselines: Random Forest and XGBoost


Evaluation: Cross-validation with metrics including R², RMSE, MAE, AARD, and Max ARD

📂 Repository Structure
├── data/                  # Input dataset (curated and raw)
│   ├── raw/              # Original experimental data
│   └── processed/        # Cleaned and feature-selected datasets
├── preprocessing/         # Scripts for descriptor generation and feature selection
│   ├── generate_descriptors.py
│   ├── data_preparation.py
│   └── feature_selection.py
├── models/                # MLP, EDL, RF, XGBoost implementations
│   ├── train_mlp.py
│   ├── train_edl.py
│   ├── train_rf_xgb.py
│   └── utils/
├── results/               # Training curves, SHAP analysis, evaluation outputs
│   ├── plots/
│   ├── metrics/
│   └── shap_analysis/
├── figures/               # Framework diagrams and plots
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
⚙️ Methodology Pipeline
1. Data Augmentation

Retrieve SMILES → RDKit 3D structure → Mordred descriptors (3,761 features)
Merge with raw experimental data (109 columns → 3,761 total features)

2. Preprocessing & Feature Selection

Cleaning: Remove constants/duplicates, impute missing values
Outlier removal: Remove 20 extreme viscosity data points
Feature reduction: Apply variance and correlation filters
Feature selection: Ensemble ranking using RF, F-score, Mutual Information, and SHAP
Final dataset: 102 descriptors across constitutional, topological, charge, autocorrelation, VSA, and physicochemical categories

3. Modeling

MLP: Multi-layer perceptron with ReLU activation, dropout regularization, Adam optimizer, and Bayesian hyperparameter tuning
EDL: Evidential Deep Learning with uncertainty quantification outputs (μ, ν, α, β parameters)
RF/XGBoost: Tree-based models as deterministic baselines

4. Evaluation

Data split: Train/Validation/Test (67%/13%/20%)
Metrics: R², RMSE, MAE, AARD (Average Absolute Relative Deviation), Max ARD
Cross-validation: Robust performance assessment

📊 Key Features

🔬 Uncertainty Quantification: EDL model provides both predictions and uncertainty estimates
🔍 Feature Interpretation: SHAP values for model explainability
🔄 Reproducible Pipeline: Fixed dataset splits and seed-based reproducibility
⚡ Hyperparameter Optimization: Optuna-based automated tuning
📈 Comprehensive Evaluation: Multiple metrics and visualization tools

🔧 Requirements

Python ≥ 3.9
RDKit
Mordred
Scikit-learn
PyTorch
Optuna
XGBoost
SHAP
Matplotlib
Pandas
NumPy

Install all dependencies:
bashpip install -r requirements.txt
🚀 Quick Start
1. Clone the Repository
bashgit clone https://github.com/yourusername/des-viscosity-prediction.git
cd des-viscosity-prediction
2. Install Dependencies
bashpip install -r requirements.txt
3. Generate Molecular Descriptors
bashpython preprocessing/generate_descriptors.py
4. Run Data Preprocessing
bashpython preprocessing/data_preparation.py
5. Train Models
Train all models:
bash# Train MLP model
python models/train_mlp.py

# Train EDL model  
python models/train_edl.py

# Train baseline models (RF & XGBoost)
python models/train_rf_xgb.py --data data/processed/final_dataset.csv
Quick test with fewer optimization trials:
bashpython models/train_rf_xgb.py --data data/processed/final_dataset.csv --trials 10
6. Evaluate and Visualize Results
bashpython results/evaluate.py
