import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_prepare_data(file_path):
    """Load and preprocess the data"""
    df = pd.read_csv(file_path)

    # Find columns
    viscosity_cols = [col for col in df.columns if 'viscosity' in col.lower()]
    if not viscosity_cols:
        raise ValueError("Viscosity column not found")
    target_col = viscosity_cols[0]

    set_cols = [col for col in df.columns if col.lower() == 'set']
    if not set_cols:
        raise ValueError("Set column not found")
    set_col = set_cols[0]

    # Get training data only
    train_data = df[df[set_col] == 'training'].copy()
    print(f"Training samples: {len(train_data)}")

    # Prepare features and target
    X_train = train_data.drop([target_col, set_col], axis=1)
    y_train = np.log1p(train_data[target_col])  

    # Preprocessing
    categorical_features = X_train.select_dtypes(include=["object"]).columns
    numerical_features = X_train.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    # Get feature names
    if len(categorical_features) > 0:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = list(ohe_feature_names) + list(numerical_features)
    else:
        all_feature_names = list(numerical_features)

    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    print(f"Feature dimension: {X_train_df.shape[1]}")

    return X_train_df, y_train

def select_features(X_train, y_train, n_features=None):
    """Select top k features if specified"""
    if n_features is None or n_features >= X_train.shape[1]:
        print(f"Using all {X_train.shape[1]} features")
        return X_train

    print(f"Selecting top {n_features} features out of {X_train.shape[1]}")
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()

    return pd.DataFrame(X_selected, columns=selected_features)

def calculate_aard_log_space(y_true_log, y_pred_log):
    """Calculate AARD in log space"""
    try:
        y_true_orig = np.exp(y_true_log)
        y_pred_orig = np.exp(y_pred_log)
        relative_deviations = np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))
        aard = np.mean(relative_deviations) * 100
        return float(aard)
    except:
        return 999.0

class ViscosityNN(nn.Module):
    """Neural Network Model"""
    def __init__(self, input_dim, hidden_layers, units, dropout_rate):
        super(ViscosityNN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(units, 1))
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)

def train_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, l2_lambda, patience=15):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    # Convert to tensors
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_val, 'values'):
        y_val = y_val.values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor).item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model

def cross_validate_model(X, y, hyperparams, n_folds=5, epochs=100):
    """Perform k-fold cross validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx].values
        X_val_fold = X.iloc[val_idx].values
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = ViscosityNN(
            input_dim=X.shape[1],
            hidden_layers=hyperparams["hidden_layers"],
            units=hyperparams["units"],
            dropout_rate=hyperparams["dropout_rate"]
        )

        try:
            model = train_model(
                model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                epochs=epochs,
                learning_rate=hyperparams["learning_rate"],
                batch_size=hyperparams["batch_size"],
                l2_lambda=hyperparams["l2_lambda"]
            )

            # Evaluate
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
                y_pred_log = model(X_val_tensor).flatten().numpy()

                if hasattr(y_val_fold, 'values'):
                    y_val_fold = y_val_fold.values

                aard = calculate_aard_log_space(y_val_fold, y_pred_log)
                cv_results.append(aard)

        except Exception as e:
            print(f"Fold {fold+1} failed: {e}")
            cv_results.append(999.0)

    return np.mean(cv_results)

def optimize_hyperparameters(X_train, y_train, n_trials=100, cv_folds=5):
    """Optimize hyperparameters using Optuna with cross-validation"""

    def objective(trial):
        hyperparams = {
            "hidden_layers": trial.suggest_int("hidden_layers", 2, 6),
            "units": trial.suggest_int("units", 64, 512, step=32),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
            "l2_lambda": trial.suggest_float("l2_lambda", 1e-6, 1e-3, log=True)
        }

        cv_aard = cross_validate_model(X_train, y_train, hyperparams, cv_folds, epochs=80)
        return cv_aard

    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value

def main(file_path, n_features=None, n_trials=100, cv_folds=5):
    """Main optimization function"""
    print(f"Starting hyperparameter optimization...")
    print(f"Features: {n_features if n_features else 'ALL'}")
    print(f"CV Folds: {cv_folds}")
    print(f"Optimization Trials: {n_trials}")
    print("="*60)

    # Load and prepare data
    X_train, y_train = load_and_prepare_data(file_path)

    # Apply feature selection if specified
    if n_features is not None:
        X_train = select_features(X_train, y_train, n_features)

    # Optimize hyperparameters
    print("\nStarting optimization...")
    best_params, best_cv_aard = optimize_hyperparameters(
        X_train, y_train, n_trials=n_trials, cv_folds=cv_folds
    )

    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best CV AARD: {best_cv_aard:.4f}%")
    print("\nBest Hyperparameters:")
    print("-" * 30)
    for param, value in best_params.items():
        print(f"{param:<15}: {value}")

    # Save results
    results = {
        'best_params': best_params,
        'best_cv_aard': best_cv_aard,
        'n_features': n_features if n_features else 'ALL',
        'cv_folds': cv_folds,
        'n_trials': n_trials
    }

    # Save to file
    feature_suffix = f"_{n_features}features" if n_features else "_allfeatures"
    filename = f'optimization_results{feature_suffix}_{cv_folds}fold_{n_trials}trials.csv'

    results_df = pd.DataFrame([{
        'Parameter': k,
        'Value': v
    } for k, v in best_params.items()] + [
        {'Parameter': 'CV_AARD', 'Value': best_cv_aard},
        {'Parameter': 'N_Features', 'Value': n_features if n_features else 'ALL'},
        {'Parameter': 'CV_Folds', 'Value': cv_folds},
        {'Parameter': 'N_Trials', 'Value': n_trials}
    ])

    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")

    return best_params, best_cv_aard

if __name__ == '__main__':
    # Update this path to your dataset
    file_path = '/kaggle/input/Viscosity_prediction_project/Final_Processed_Data.csv'

    # Run optimization
    best_params, best_aard = main(
        file_path=file_path,
        n_features=None,    
        n_trials=150,      
        cv_folds=5         
    )

