%pip install xgboost optuna

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Data preparation function 
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Find the viscosity column (case-insensitive)
    viscosity_cols = [col for col in df.columns if 'viscosity' in col.lower()]
    if not viscosity_cols:
        raise ValueError("Viscosity column not found in the dataset")

    target_col = viscosity_cols[0]
    print(f"Using target column: {target_col}")

    # Find the Set column (case-insensitive)
    set_cols = [col for col in df.columns if col.lower() == 'set']
    if not set_cols:
        raise ValueError("Set column not found in the dataset")

    set_col = set_cols[0]
    print(f"Using set column: {set_col}")

    # Separate training and test data based on Set column
    train_data = df[df[set_col] == 'training'].copy()
    test_data = df[df[set_col] == 'test'].copy()

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Prepare features and target for training data
    X_train_full = train_data.drop([target_col, set_col], axis=1)
    y_train_full = np.log1p(train_data[target_col])  

    # Prepare features and target for test data
    X_test = test_data.drop([target_col, set_col], axis=1)
    y_test = np.log1p(test_data[target_col])  

    # Split training data into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"Final train samples: {len(X_train)}")
    print(f"Final validation samples: {len(X_val)}")

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=["object"]).columns
    numerical_features = X_train.select_dtypes(exclude=["object"]).columns

    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")

    # Preprocessing pipeline 
    if len(categorical_features) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
                ("num", "passthrough", numerical_features)  
            ]
        )

        # Fit preprocessor on training data only
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = list(ohe_feature_names) + list(numerical_features)
    else:
        # No categorical features, use data as is
        X_train_processed = X_train.values
        X_val_processed = X_val.values
        X_test_processed = X_test.values
        all_feature_names = list(numerical_features)
        preprocessor = None

    # Create DataFrames with processed features
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_val_df = pd.DataFrame(X_val_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    print(f"Final feature dimension: {X_train_df.shape[1]}")

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, preprocessor

def calculate_aard_log_space(y_true_log, y_pred_log):
    """Calculate AARD when data is already in log space"""
    try:
        # Convert back to original space for AARD calculation
        y_true_orig = np.exp(y_true_log)
        y_pred_orig = np.exp(y_pred_log)

        # Calculate AARD in original space
        relative_deviations = np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))
        aard = np.mean(relative_deviations) * 100
        max_ard = np.max(relative_deviations) * 100

        return float(aard), float(max_ard)
    except:
        return 999.0, 999.0

# Function to evaluate the model with all requested metrics
def evaluate_model(model, X, y, model_name="Model"):
    y_pred_log = model.predict(X)

    # Calculate AARD and MAX ARD using the same function as MLP
    aard, max_ard = calculate_aard_log_space(y.values, y_pred_log)

    # Calculate other metrics in log scale
    r2_log = r2_score(y.values, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y.values, y_pred_log))
    mae_log = mean_absolute_error(y.values, y_pred_log)

    # Convert back to original scale for interpretation
    y_original = np.expm1(y.values)
    y_pred_original = np.expm1(y_pred_log)

    # Calculate metrics in original scale for comparison
    r2_orig = r2_score(y_original, y_pred_original)
    rmse_orig = np.sqrt(mean_squared_error(y_original, y_pred_original))
    mae_orig = mean_absolute_error(y_original, y_pred_original)

    return {
        'r2_log': r2_log,
        'rmse_log': rmse_log,
        'mae_log': mae_log,
        'aard': aard,  
        'max_ard': max_ard,  
        'r2_orig': r2_orig,
        'rmse_orig': rmse_orig,
        'mae_orig': mae_orig,
        'y_pred_log': y_pred_log,
        'y_pred_orig': y_pred_original
    }

# Random Forest optimization
def optimize_random_forest(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        results = evaluate_model(model, X_val, y_val)
        return results['aard']  # Minimize AARD

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    return study.best_params, study.best_value

# XGBoost optimization
def optimize_xgboost(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'random_state': 42
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20, verbose=False)

        results = evaluate_model(model, X_val, y_val)
        return results['aard']  # Minimize AARD

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    return study.best_params, study.best_value

# Visualization function
def plot_results(y_true, y_pred, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Parity plot
    ax1.scatter(y_true, y_pred, alpha=0.6)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Viscosity')
    ax1.set_ylabel('Predicted Viscosity')
    ax1.set_title(f'{title} - Parity Plot')
    ax1.grid(True, alpha=0.3)

    # Residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Viscosity')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{title} - Residual Plot')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main(file_path):
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_prepare_data(file_path)

    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")

    models = {}
    results_summary = []

    # 1. Random Forest
    print("\n" + "="*60)
    print("OPTIMIZING RANDOM FOREST")
    print("="*60)
    rf_best_params, rf_best_aard = optimize_random_forest(X_train, y_train, X_val, y_val)
    print(f"Best RF AARD: {rf_best_aard:.3f}%")
    print("Best RF Parameters:", rf_best_params)

    # Train final RF model
    rf_model = RandomForestRegressor(**rf_best_params)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model

    # Evaluate RF
    rf_train = evaluate_model(rf_model, X_train, y_train)
    rf_val = evaluate_model(rf_model, X_val, y_val)
    rf_test = evaluate_model(rf_model, X_test, y_test)

    results_summary.append({
        'Model': 'Random Forest',
        'Train_AARD': rf_train['aard'],
        'Val_AARD': rf_val['aard'],
        'Test_AARD': rf_test['aard'],
        'Test_MaxARD': rf_test['max_ard'],
        'Test_R2': rf_test['r2_log']
    })

    # 2. XGBoost
    print("\n" + "="*60)
    print("OPTIMIZING XGBOOST")
    print("="*60)
    xgb_best_params, xgb_best_aard = optimize_xgboost(X_train, y_train, X_val, y_val)
    print(f"Best XGB AARD: {xgb_best_aard:.3f}%")
    print("Best XGB Parameters:", xgb_best_params)

    # Train final XGB model
    xgb_model = xgb.XGBRegressor(**xgb_best_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20, verbose=False)
    models['XGBoost'] = xgb_model

    # Evaluate XGB
    xgb_train = evaluate_model(xgb_model, X_train, y_train)
    xgb_val = evaluate_model(xgb_model, X_val, y_val)
    xgb_test = evaluate_model(xgb_model, X_test, y_test)

    results_summary.append({
        'Model': 'XGBoost',
        'Train_AARD': xgb_train['aard'],
        'Val_AARD': xgb_val['aard'],
        'Test_AARD': xgb_test['aard'],
        'Test_MaxARD': xgb_test['max_ard'],
        'Test_R2': xgb_test['r2_log']
    })

    # Print comprehensive results
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)

    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))

    # Detailed results for each model
    all_results = {
        'Random Forest': (rf_train, rf_val, rf_test),
        'XGBoost': (xgb_train, xgb_val, xgb_test)
    }

    print("\n" + "="*80)
    print("DETAILED METRICS FOR ALL MODELS")
    print("="*80)

    for model_name, (train_res, val_res, test_res) in all_results.items():
        print(f"\n{model_name}:")
        print(f"{'Dataset':<12} {'AARD%':<8} {'MaxARD%':<8} {'R²':<8} {'RMSE':<8} {'MAE':<8}")
        print("-" * 60)
        print(f"{'Train':<12} {train_res['aard']:<8.3f} {train_res['max_ard']:<8.3f} {train_res['r2_log']:<8.4f} {train_res['rmse_log']:<8.4f} {train_res['mae_log']:<8.4f}")
        print(f"{'Val':<12} {val_res['aard']:<8.3f} {val_res['max_ard']:<8.3f} {val_res['r2_log']:<8.4f} {val_res['rmse_log']:<8.4f} {val_res['mae_log']:<8.4f}")
        print(f"{'Test':<12} {test_res['aard']:<8.3f} {test_res['max_ard']:<8.3f} {test_res['r2_log']:<8.4f} {test_res['rmse_log']:<8.4f} {test_res['mae_log']:<8.4f}")

    # Find best model based on test AARD
    best_model_name = min(results_summary, key=lambda x: x['Test_AARD'])['Model']
    print(f"\nBest Model: {best_model_name} (Test AARD: {min(results_summary, key=lambda x: x['Test_AARD'])['Test_AARD']:.3f}%)")

    # Plot results for the best model
    best_train, best_val, best_test = all_results[best_model_name]

    y_train_orig = np.expm1(y_train.values)
    y_val_orig = np.expm1(y_val.values)
    y_test_orig = np.expm1(y_test.values)

    plot_results(y_train_orig, best_train['y_pred_orig'], f"{best_model_name} - Training Set")
    plot_results(y_val_orig, best_val['y_pred_orig'], f"{best_model_name} - Validation Set")
    plot_results(y_test_orig, best_test['y_pred_orig'], f"{best_model_name} - Test Set")

    # Save results
    results_df.to_csv('tree_models_results.csv', index=False)

    # Create a comprehensive comparison DataFrame
    detailed_results = []
    for model_name, (train_res, val_res, test_res) in all_results.items():
        for dataset, res in zip(['Train', 'Val', 'Test'], [train_res, val_res, test_res]):
            detailed_results.append({
                'Model': model_name,
                'Dataset': dataset,
                'AARD_%': res['aard'],
                'MaxARD_%': res['max_ard'],
                'R²_log': res['r2_log'],
                'RMSE_log': res['rmse_log'],
                'MAE_log': res['mae_log'],
                'R²_orig': res['r2_orig'],
                'RMSE_orig': res['rmse_orig'],
                'MAE_orig': res['mae_orig']
            })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('detailed_tree_models_results.csv', index=False)

    print("\nResults saved to 'tree_models_results.csv' and 'detailed_tree_models_results.csv'")

    return models, results_df, detailed_df

if __name__ == '__main__':
    # Update this path to your dataset
    file_path = '/content/Final_Processed_Data.csv'  
    models, results, detailed_results = main(file_path)