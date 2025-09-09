import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Updated matplotlib style configuration 
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'Times New Roman',  
    'axes.linewidth': 1.2,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Feature selection function 
def select_features(X_train, y_train, X_val, X_test, n_features=None):
    """
    Select top k features based on univariate statistical tests
    If n_features is None, use all features
    """
    if n_features is None or n_features >= X_train.shape[1]:
        print(f"Using all {X_train.shape[1]} features")
        return X_train, X_val, X_test, list(X_train.columns)

    print(f"Selecting top {n_features} features out of {X_train.shape[1]}")

    # Fit feature selector on training data
    selector = SelectKBest(score_func=f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features[:5]}..." if len(selected_features) > 5 else f"Selected features: {selected_features}")

    # Convert back to DataFrames
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_features)
    X_val_df = pd.DataFrame(X_val_selected, columns=selected_features)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_features)

    return X_train_df, X_val_df, X_test_df, selected_features

# Data preparation function 
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Find the viscosity column 
    viscosity_cols = [col for col in df.columns if 'viscosity' in col.lower()]
    if not viscosity_cols:
        raise ValueError("Viscosity column not found in the dataset")

    target_col = viscosity_cols[0]
    print(f"Using target column: {target_col}")

    # Find the Set column 
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

    print(f"Final train samples: {len(X_train_full)}")

    # Identify categorical and numerical features
    categorical_features = X_train_full.select_dtypes(include=["object"]).columns
    numerical_features = X_train_full.select_dtypes(exclude=["object"]).columns

    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")

    # Preprocessing pipeline - fit only on training data
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )

    # Fit preprocessor on training data only
    X_train_processed = preprocessor.fit_transform(X_train_full)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    if len(categorical_features) > 0:
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = list(ohe_feature_names) + list(numerical_features)
    else:
        all_feature_names = list(numerical_features)

    # Create DataFrames with processed features
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    print(f"Final feature dimension: {X_train_df.shape[1]}")

    return X_train_df, X_test_df, y_train_full, y_test, preprocessor, all_feature_names

def calculate_aard_log_space(y_true_log, y_pred_log):
    """Calculate AARD in log space"""
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

# Define the PyTorch neural network model
class ViscosityNN(nn.Module):
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)

# Function to train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size=32, patience=20, l2_lambda=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Convert to numpy arrays if they are pandas Series
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_val, 'values'):
        y_val = y_val.values

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor).item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model, train_losses, val_losses

# Function to evaluate the model
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred_log = model(X_tensor).flatten().numpy()

        # Convert y to numpy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values

        # Calculate AARD and MAX ARD using your function
        aard, max_ard = calculate_aard_log_space(y, y_pred_log)

        # Calculate metrics in log scale
        r2 = r2_score(y, y_pred_log)
        rmse = np.sqrt(mean_squared_error(y, y_pred_log))
        mae = mean_absolute_error(y, y_pred_log)

        # Convert back to original scale
        y_original = np.expm1(y)
        y_pred_original = np.expm1(y_pred_log)

        # Calculate metrics in original scale
        r2_orig = r2_score(y_original, y_pred_original)
        rmse_orig = np.sqrt(mean_squared_error(y_original, y_pred_original))
        mae_orig = mean_absolute_error(y_original, y_pred_original)

        # Calculate absolute relative deviations
        relative_deviations = np.abs((y_original - y_pred_original) / (y_original + 1e-8))

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'aard': aard,
        'max_ard': max_ard,
        'r2_orig': r2_orig,
        'rmse_orig': rmse_orig,
        'mae_orig': mae_orig,
        'y_pred_log': y_pred_log,
        'y_pred_orig': y_pred_original,
        'y_true_orig': y_original,
        'relative_deviations': relative_deviations
    }

def create_cv_plots(all_fold_results, feature_suffix=""):
    """Create comprehensive cross-validation plots """

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Set Times New Roman font for this figure
    plt.rcParams.update({'font.family': 'Times New Roman'})

    # Collect all data for combined plots
    all_train_true, all_train_pred = [], []
    all_val_true, all_val_pred = [], []
    all_test_true, all_test_pred = [], []

    for i, fold_data in enumerate(all_fold_results):
        all_train_true.extend(fold_data['train_results']['y_true_orig'])
        all_train_pred.extend(fold_data['train_results']['y_pred_orig'])
        all_val_true.extend(fold_data['val_results']['y_true_orig'])
        all_val_pred.extend(fold_data['val_results']['y_pred_orig'])
        all_test_true.extend(fold_data['test_results']['y_true_orig'])
        all_test_pred.extend(fold_data['test_results']['y_pred_orig'])

    all_train_true, all_train_pred = np.array(all_train_true), np.array(all_train_pred)
    all_val_true, all_val_pred = np.array(all_val_true), np.array(all_val_pred)
    all_test_true, all_test_pred = np.array(all_test_true), np.array(all_test_pred)

    # Calculate combined R²
    train_r2_combined = r2_score(all_train_true, all_train_pred)
    val_r2_combined = r2_score(all_val_true, all_val_pred)
    test_r2_combined = r2_score(all_test_true, all_test_pred)

    # Calculate combined AARD (this gives the "combined" result)
    train_aard_combined = np.mean(np.abs((all_train_true - all_train_pred) / (all_train_true + 1e-8))) * 100
    val_aard_combined = np.mean(np.abs((all_val_true - all_val_pred) / (all_val_true + 1e-8))) * 100
    test_aard_combined = np.mean(np.abs((all_test_true - all_test_pred) / (all_test_true + 1e-8))) * 100

    # Calculate mean of fold AARDs (this should match your table)
    fold_train_aards = [fold_data['train_results']['aard'] for fold_data in all_fold_results]
    fold_val_aards = [fold_data['val_results']['aard'] for fold_data in all_fold_results]
    fold_test_aards = [fold_data['test_results']['aard'] for fold_data in all_fold_results]

    mean_train_aard = np.mean(fold_train_aards)
    mean_val_aard = np.mean(fold_val_aards)
    mean_test_aard = np.mean(fold_test_aards)

    print(f"\nAArd Calculation Comparison:")
    print(f"Train - Combined data: {train_aard_combined:.2f}%, Mean of folds: {mean_train_aard:.2f}%")
    print(f"Val   - Combined data: {val_aard_combined:.2f}%, Mean of folds: {mean_val_aard:.2f}%")
    print(f"Test  - Combined data: {test_aard_combined:.2f}%, Mean of folds: {mean_test_aard:.2f}%")

    # Training set plots - Use mean of folds for consistency
    axes[0,0].scatter(all_train_true, all_train_pred, alpha=0.6, color='blue', s=20)
    axes[0,0].plot([all_train_true.min(), all_train_true.max()], [all_train_true.min(), all_train_true.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[0,0].set_ylabel('Predicted Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[0,0].grid(True, alpha=0.3)

    train_residuals = all_train_true - all_train_pred
    axes[1,0].scatter(all_train_pred, train_residuals, alpha=0.6, color='blue', s=20)
    axes[1,0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1,0].set_xlabel('Predicted Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[1,0].set_ylabel('Residuals (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[1,0].grid(True, alpha=0.3)

    # Validation set plots - Use mean of folds for consistency
    axes[0,1].scatter(all_val_true, all_val_pred, alpha=0.6, color='green', s=20)
    axes[0,1].plot([all_val_true.min(), all_val_true.max()], [all_val_true.min(), all_val_true.max()], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[0,1].set_ylabel('Predicted Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[0,1].grid(True, alpha=0.3)

    val_residuals = all_val_true - all_val_pred
    axes[1,1].scatter(all_val_pred, val_residuals, alpha=0.6, color='green', s=20)
    axes[1,1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1,1].set_xlabel('Predicted Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[1,1].set_ylabel('Residuals (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[1,1].grid(True, alpha=0.3)

    # Test set plots - Use mean of folds for consistency
    axes[0,2].scatter(all_test_true, all_test_pred, alpha=0.6, color='orange', s=20)
    axes[0,2].plot([all_test_true.min(), all_test_true.max()], [all_test_true.min(), all_test_true.max()], 'r--', lw=2)
    axes[0,2].set_xlabel('Actual Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[0,2].set_ylabel('Predicted Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[0,2].grid(True, alpha=0.3)

    test_residuals = all_test_true - all_test_pred
    axes[1,2].scatter(all_test_pred, test_residuals, alpha=0.6, color='orange', s=20)
    axes[1,2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1,2].set_xlabel('Predicted Viscosity (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[1,2].set_ylabel('Residuals (mPa·s)', fontweight='bold', fontfamily='Times New Roman')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'cv_parity_residual_plots{feature_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

def add_categorized_plot_to_existing_analysis(feature_importance_df, n_top_features=30, feature_suffix=""):
    """
     Categorized plot for SHAP analysis 
    """

    # Set Times New Roman font
    plt.rcParams.update({'font.family': 'Times New Roman'})

    # Complete feature categories mapping 
    feature_categories = {
        # Drug-like indices
        'MW_HBA': 'Drug-like indices', 'MwHBA': 'Drug-like indices', 'MwHBA_': 'Drug-like indices',
        'X_HBD': 'Drug-like indices', 'XHBD': 'Drug-like indices', 'XHBD_': 'Drug-like indices',
        'X_HBA': 'Drug-like indices', 'XHBA': 'Drug-like indices', 'XHBA_': 'Drug-like indices',
        'MW_HBD': 'Drug-like indices', 'MwHBD': 'Drug-like indices',

        # Topological indices
        'g5': 'Topological indices', 'g3': 'Topological indices',

        # Constitutional indices
        'n_n': 'Constitutional indices',
        'maxsssCH_HBD': 'Constitutional indices', 'hbd_maxsssCH_encoded': 'Constitutional indices',
        'ch': 'Constitutional indices', 'ch_2': 'Constitutional indices',
        'cl_cl': 'Constitutional indices',
        'nhbdon_HBD': 'Constitutional indices', 'hbd_nhbdon': 'Constitutional indices',
        'maxssch2_HBD': 'Constitutional indices',
        'hbd_gats6d_encoded': 'Constitutional indices',
        'hbd_aeta_alpha_encoded': 'Constitutional indices',
        'hbd_aats0dv': 'Constitutional indices',
        'hbd_eta_psi_1_encoded': 'Constitutional indices',
        'hbd_eta_epsilon_1_encoded': 'Constitutional indices',
        'hbd_amid_h_encoded': 'Constitutional indices',

        # Charge descriptors
        'pnsa3_HBD': 'Charge descriptors', 'hbd_pnsa3': 'Charge descriptors',
        'dpsa3_HBD': 'Charge descriptors', 'hbd_dpsa3': 'Charge descriptors',
        'fnsa3_HBD': 'Charge descriptors', 'hbd_fnsa3': 'Charge descriptors',
        'pnsa5_HBD': 'Charge descriptors',
        'rasa_HBD': 'Charge descriptors',
        'rpsa_HBD': 'Charge descriptors', 'hbd_rpsa': 'Charge descriptors',

        # VSA-like descriptors
        'peoe_vsa1_HBD': 'VSA-like descriptors', 'hbd_peoe_vsa1': 'VSA-like descriptors',
        'peoe_vsa6_HBD': 'VSA-like descriptors', 'hbd_peoe_vsa6': 'VSA-like descriptors',
        'peoe_vsa10_HBD': 'VSA-like descriptors',
        'slogp_vsa2_HBD': 'VSA-like descriptors', 'hbd_slogp_vsa2': 'VSA-like descriptors',
        'smr_vsa1_HBD': 'VSA-like descriptors', 'hbd_smr_vsa1': 'VSA-like descriptors',
        'hbd_vsa_estate7': 'VSA-like descriptors',

        # Geometrical descriptors
        'axp_4dv_HBD': 'Geometrical descriptors', 'hbd_axp_1dv': 'Geometrical descriptors',

        # Physicochemical properties
        'slogp_HBD': 'Physicochemical properties', 'hbd_slogp': 'Physicochemical properties',
        'mpe_HBD': 'Physicochemical properties',
        'mare_HBD': 'Physicochemical properties', 'hbd_mare': 'Physicochemical properties',
        't_k': 'Physicochemical properties',

        # Add more specific mappings for your actual features
        'gats6d_HBD': 'Constitutional indices',
        'maxssssCH_HBD': 'Constitutional indices',
        'maxssch2_HBD': 'Constitutional indices'
    }

    # Filter out temperature features FIRST
    exclude_features = ['Temperature', 'temperature', 'temp', 'T', 't_k']
    non_temp_mask = np.array([not any(excl.lower() in feat.lower() for excl in exclude_features)
                             for feat in feature_importance_df['feature']])

    # Get top N non-temperature features
    feature_importance_no_temp = feature_importance_df[non_temp_mask]
    top_features = feature_importance_no_temp.head(n_top_features).copy()

    # Add category column and handle unmapped features more specifically
    top_features['category'] = top_features['feature'].map(feature_categories)

    # For unmapped features, try to infer category from common patterns
    for idx, row in top_features.iterrows():
        if pd.isna(row['category']):
            feature_name = row['feature'].lower()
            if any(x in feature_name for x in ['hba', 'hbd', 'mw_']):
                top_features.at[idx, 'category'] = 'Drug-like indices'
            elif any(x in feature_name for x in ['pnsa', 'dpsa', 'fnsa', 'rasa', 'rpsa']):
                top_features.at[idx, 'category'] = 'Charge descriptors'
            elif any(x in feature_name for x in ['vsa', 'peoe', 'smr']):
                top_features.at[idx, 'category'] = 'VSA-like descriptors'
            elif any(x in feature_name for x in ['axp', 'geomet']):
                top_features.at[idx, 'category'] = 'Geometrical descriptors'
            elif any(x in feature_name for x in ['slogp', 'mare', 'mpe']):
                top_features.at[idx, 'category'] = 'Physicochemical properties'
            elif any(x in feature_name for x in ['ch_', 'cl_', 'encoded', 'aats', 'gats']):
                top_features.at[idx, 'category'] = 'Constitutional indices'
            elif any(x in feature_name for x in ['g3', 'g5']):
                top_features.at[idx, 'category'] = 'Topological indices'
            else:
                top_features.at[idx, 'category'] = 'Constitutional indices'  

    # Define professional color palette 
    color_palette = {
        'Drug-like indices': '#4472C4',           # Blue
        'Topological indices': '#E46C0A',         # Orange
        'Constitutional indices': '#70AD47',       # Green
        'Charge descriptors': '#C55A5A',          # Red
        'VSA-like descriptors': '#9A6FB0',        # Purple
        'Geometrical descriptors': '#997300',     # Brown
        'Physicochemical properties': '#E377C2'   # Pink
    }

    # Create the plot with proper sizing 
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create horizontal bars
    y_positions = range(len(top_features))
    bars = ax.barh(y_positions, top_features['importance'],
                   color=[color_palette.get(cat, '#7f7f7f') for cat in top_features['category']],
                   alpha=0.8, edgecolor='white', linewidth=0.5)

    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features['feature'], fontsize=11, fontfamily='Times New Roman')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Feature Importance with Category and SHAP Value', fontsize=16, fontweight='bold', pad=20, fontfamily='Times New Roman')

    # Add SHAP values at the end of bars 
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'] + max(top_features['importance']) * 0.01, i,
                f'{row["importance"]:.3f}', 
                va='center', ha='left', fontsize=10, fontweight='bold', fontfamily='Times New Roman')

    # Invert y-axis so highest importance is at top
    ax.invert_yaxis()

    # Add grid
    ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Create legend with all categories represented in data
    legend_elements = []
    displayed_categories = sorted(top_features['category'].unique())
    for category in displayed_categories:
        color = color_palette.get(category, '#7f7f7f')
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, label=category))

    # Position legend at the bottom right within the plot area
    legend = ax.legend(handles=legend_elements, title='Category',
                      loc='lower right',
                      fontsize=12, title_fontsize=11, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.95,
                      prop={'family': 'Times New Roman'})

    # Style the legend title
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontfamily('Times New Roman')

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Set tick label fonts
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'shap_categorized_importance{feature_suffix}.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

    print(f"Professor's preferred categorized SHAP plot saved as: shap_categorized_importance{feature_suffix}.png")

    return fig, ax

def create_shap_analysis(final_model, X_train, feature_names, n_top_features=10, feature_suffix=""):
    """Create professional SHAP analysis plots"""

    # Set Times New Roman font for all plots
    plt.rcParams.update({'font.family': 'Times New Roman'})

    print(f"\nGenerating SHAP analysis for top {n_top_features} features (CORRECTED VERSION)...")

    # DON'T filter features 
    print(f"Using all {len(feature_names)} features (model was trained on this)")

    # Optimized sample sizes for speed while maintaining quality
    sample_size = 1000
    background_size = 50

    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    background_indices = np.random.choice(len(X_train), background_size, replace=False)

    X_sample = X_train[sample_indices]
    X_background = X_train[background_indices]

    # Convert to tensors
    X_background_tensor = torch.tensor(X_background.astype(np.float32), dtype=torch.float32)
    X_sample_tensor = torch.tensor(X_sample.astype(np.float32), dtype=torch.float32)

    print("Using DeepExplainer for neural networks...")

    # Use DeepExplainer 
    explainer = shap.DeepExplainer(final_model, X_background_tensor)
    shap_values = explainer.shap_values(X_sample_tensor)

    # Handle different return formats
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Convert to numpy if still tensor
    if hasattr(shap_values, 'detach'):
        shap_values_np = shap_values.detach().numpy()
    else:
        shap_values_np = shap_values

    # Ensure 2D array
    if shap_values_np.ndim == 3:
        shap_values_np = shap_values_np.reshape(shap_values_np.shape[0], -1)

    print(f"SHAP values computed successfully! Shape: {shap_values_np.shape}")

    # Get feature importance for ALL features
    feature_importance = np.abs(shap_values_np).mean(0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Filter out temperature features
    exclude_features = ['Temperature', 'temperature', 'temp', 'T', 't_k']
    non_temp_mask = np.array([not any(excl.lower() in feat.lower() for excl in exclude_features)
                              for feat in feature_importance_df['feature']])

    # Get non-temperature features for display
    feature_importance_no_temp = feature_importance_df[non_temp_mask].head(n_top_features)

    print(f"Displaying top {n_top_features} non-temperature features out of {len(feature_names)} total features")

    # Get indices in original feature space for top non-temperature features
    top_feature_names = feature_importance_no_temp['feature'].values
    top_features_idx = [list(feature_names).index(fname) for fname in top_feature_names]

    # Extract SHAP values and sample data for top features only for visualization
    shap_values_top = shap_values_np[:, top_features_idx]
    X_sample_top = X_sample[:, top_features_idx]

    print(f"Creating journal-quality SHAP plots for top {n_top_features} features...")

    # 1. SHAP Summary Plot (Beeswarm) WITH VALUES 
    fig, ax = plt.subplots(figsize=(14, 8))  
    shap.summary_plot(shap_values_top, X_sample_top,
                      feature_names=top_feature_names, show=False, max_display=n_top_features,
                      color_bar=True, cmap='RdYlBu_r')

    # Get current axis limits
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate text position in data coordinates
    text_x_position = xlim[1] + (xlim[1] - xlim[0]) * 0.15  

    display_features = min(n_top_features, len(top_feature_names))
    for i in range(display_features):
        mean_shap = np.mean(np.abs(shap_values_top[:, i]))
        y_pos = display_features - 1 - i
        ax.text(text_x_position, y_pos, f'{mean_shap:.3f}',
                va='center', ha='left',
                fontsize=10, fontweight='bold', fontfamily='Times New Roman',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

    # Add header for the values column
    ax.text(text_x_position, display_features + 0.5, 'Mean |SHAP|',
            va='center', ha='left',
            fontsize=12, fontweight='bold', fontfamily='Times New Roman',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, edgecolor='none'))

    # Extend x-axis limit to accommodate the text
    new_xlim_right = text_x_position + (xlim[1] - xlim[0]) * 0.12
    ax.set_xlim(xlim[0], new_xlim_right)

    plt.xlabel('SHAP Value (Impact on Viscosity Prediction)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.tick_params(labelsize=12)

    # Set font for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.savefig(f'shap_summary_plot{feature_suffix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 2. SHAP Bar Plot (Feature Importance) 
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_top, X_sample_top,
                      feature_names=top_feature_names, plot_type="bar", show=False,
                      max_display=n_top_features, color='steelblue')

    ax = plt.gca()
    max_importance = feature_importance_no_temp['importance'].max()

    for i in range(min(n_top_features, len(top_feature_names))):
        importance_val = feature_importance_no_temp.iloc[i]['importance']
        y_pos = len(top_feature_names) - 1 - i
        ax.text(importance_val + max_importance * 0.02, y_pos, f'{importance_val:.3f}',
                va='center', ha='left', fontsize=9, fontweight='bold', fontfamily='Times New Roman')

    plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.tick_params(labelsize=10)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'shap_bar_plot{feature_suffix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 3. SHAP Waterfall Plots - CORRECTED
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    exp_value = explainer.expected_value
    if isinstance(exp_value, (list, np.ndarray, torch.Tensor)):
        if hasattr(exp_value, 'detach'):
            exp_value = exp_value.detach().numpy()
        exp_value = float(exp_value[0]) if len(exp_value) > 0 else 0.0
    else:
        exp_value = float(exp_value)

    for i in range(4):
        plt.sca(axes[i])
        sample_idx = i * (len(shap_values_top) // 4)

        explanation = shap.Explanation(
            values=shap_values_top[sample_idx],
            base_values=exp_value,
            data=X_sample_top[sample_idx],
            feature_names=top_feature_names[:len(shap_values_top[sample_idx])]
        )

        try:
            shap.plots.waterfall(explanation, max_display=15, show=False)
        except:
            shap.waterfall_plot(explanation, max_display=15, show=False)

        plt.title(f'Sample {i+1}', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
        axes[i].tick_params(labelsize=10)

        for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
            label.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.savefig(f'shap_waterfall_plots{feature_suffix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 4. SHAP Decision Plot for Representative Samples
    fig, ax = plt.subplots(figsize=(12, 8))

    predictions = final_model(X_sample_tensor[:len(shap_values_top)]).detach().numpy().flatten()

    high_pred_idx = np.where(predictions >= np.percentile(predictions, 80))[0]
    med_pred_idx = np.where((predictions >= np.percentile(predictions, 40)) &
                            (predictions <= np.percentile(predictions, 60)))[0]
    low_pred_idx = np.where(predictions <= np.percentile(predictions, 20))[0]

    n_samples_per_group = 3
    selected_indices = []
    if len(high_pred_idx) > 0:
        selected_indices.extend(np.random.choice(high_pred_idx, min(n_samples_per_group, len(high_pred_idx)), replace=False))
    if len(med_pred_idx) > 0:
        selected_indices.extend(np.random.choice(med_pred_idx, min(n_samples_per_group, len(med_pred_idx)), replace=False))
    if len(low_pred_idx) > 0:
        selected_indices.extend(np.random.choice(low_pred_idx, min(n_samples_per_group, len(low_pred_idx)), replace=False))

    n_enhanced_features = min(10, len(top_feature_names))

    exp_value = explainer.expected_value
    if isinstance(exp_value, (list, np.ndarray)):
        exp_value = float(exp_value[0]) if len(exp_value) > 0 else 0.0
    else:
        exp_value = float(exp_value)

    shap.decision_plot(exp_value,
                       shap_values_top[selected_indices, :n_enhanced_features],
                       X_sample_top[selected_indices, :n_enhanced_features],
                       feature_names=top_feature_names[:n_enhanced_features],
                       show=False,
                       feature_order='importance',
                       highlight=[0, 3, 6])

    plt.title(f'SHAP Decision Plot - Top {n_enhanced_features} Features\n' +
              'Prediction Paths (High/Med/Low)',
              fontsize=14, fontweight='bold', pad=15, fontfamily='Times New Roman')
    plt.xlabel('Model Output (Viscosity)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.tick_params(labelsize=12)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.savefig(f'shap_decision_plot{feature_suffix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 5. SHAP Dependence Plots for top 6 most important features 
    if len(top_feature_names) >= 6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i in range(6):
            plt.sca(axes[i])
            feature_idx = i

            # Find best interaction feature
            interactions = []
            for j in range(len(top_feature_names)):
                if j != feature_idx:
                    try:
                        corr = np.corrcoef(shap_values_top[:, feature_idx], X_sample_top[:, j])[0,1]
                        if not np.isnan(corr):
                            interactions.append((j, abs(corr)))
                    except:
                        continue

            interaction_idx = max(interactions, key=lambda x: x[1])[0] if interactions else None

            shap.dependence_plot(feature_idx, shap_values_top, X_sample_top,
                               feature_names=top_feature_names,
                               interaction_index=interaction_idx,
                               show=False, color='darkslateblue', alpha=0.6)
            plt.title('')
            plt.xlabel(top_feature_names[feature_idx], fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            plt.ylabel('SHAP Value', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
            axes[i].tick_params(labelsize=10)

            # Set font for tick labels
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontfamily('Times New Roman')

        plt.tight_layout()
        plt.savefig(f'shap_dependence_plots{feature_suffix}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()

    # 6. Top Features Analysis Table 
    print(f"\n{'='*60}")
    print(f"TOP {n_top_features} MOST IMPORTANT NON-TEMPERATURE FEATURES")
    print(f"{'='*60}")
    print(f"{'Rank':<4} {'Feature':<40} {'Importance':<12} {'Mean SHAP':<12} {'Std SHAP':<12}")
    print("-" * 82)

    detailed_importance = []
    for i, (_, row) in enumerate(feature_importance_no_temp.head(n_top_features).iterrows(), 1):
        try:
            feature_idx = list(feature_names).index(row['feature'])
            feature_shap_values = shap_values_np[:, feature_idx]

            mean_shap = np.mean(feature_shap_values)
            std_shap = np.std(feature_shap_values)

            print(f"{i:<4} {row['feature']:<40} {row['importance']:<12.3f} {mean_shap:<12.3f} {std_shap:<12.3f}")  

            detailed_importance.append({
                'rank': i,
                'feature': row['feature'],
                'mean_abs_shap': row['importance'],
                'mean_shap': mean_shap,
                'std_shap': std_shap,
                'min_shap': np.min(feature_shap_values),
                'max_shap': np.max(feature_shap_values),
                'positive_impact_pct': (feature_shap_values > 0).mean() * 100
            })
        except Exception as e:
            print(f"Error processing feature {row['feature']}: {e}")
            continue

    # Create and save detailed importance DataFrame
    detailed_importance_df = pd.DataFrame(detailed_importance)

    # Save results 
    feature_importance_no_temp.head(n_top_features).to_csv(f'shap_feature_importance{feature_suffix}.csv', index=False)
    detailed_importance_df.to_csv(f'shap_detailed_analysis{feature_suffix}.csv', index=False)

    # Also save ALL features for reference 
    feature_importance_df[non_temp_mask].to_csv(f'shap_all_features_importance{feature_suffix}.csv', index=False)

    # 7. Feature Importance Visualization (Top 15 features) 
    fig, ax = plt.subplots(figsize=(10, 6))
    top_15 = detailed_importance_df.head(15)

    bars = plt.barh(range(len(top_15)), top_15['mean_abs_shap'], color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_15)), top_15['feature'], fontsize=10, fontfamily='Times New Roman')
    plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('Features', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold', fontfamily='Times New Roman')

    # Add value labels on bars 
    max_val = top_15['mean_abs_shap'].max()
    for i, (idx, row) in enumerate(top_15.iterrows()):
        plt.text(row['mean_abs_shap'] + max_val * 0.02, i,
                f'{row["mean_abs_shap"]:.3f}',  # Changed to 3 decimal places
                va='center', fontsize=9, fontweight='bold', ha='left', fontfamily='Times New Roman')

    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Set font for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    plt.savefig(f'shap_top15_importance{feature_suffix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 8. SHAP Force Plot for individual predictions 
    try:
        print("Creating SHAP force plots for individual samples...")

        # Select 3 diverse samples (low, medium, high predictions)
        predictions = np.sum(shap_values_top, axis=1) + exp_value
        sorted_indices = np.argsort(predictions)

        sample_indices_force = [
            sorted_indices[len(sorted_indices)//4],
            sorted_indices[len(sorted_indices)//2],
            sorted_indices[3*len(sorted_indices)//4]
        ]

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        for i, sample_idx in enumerate(sample_indices_force):
            plt.sca(axes[i])

            # Create force plot data
            shap.force_plot(
                exp_value,
                shap_values_top[sample_idx],
                X_sample_top[sample_idx],
                feature_names=top_feature_names,
                matplotlib=True,
                show=False,
                text_rotation=45
            )

            pred_value = predictions[sample_idx]
            plt.title(f'Sample {i+1}: Predicted Log Viscosity = {pred_value:.3f}',  
                     fontsize=12, fontweight='bold', fontfamily='Times New Roman')

            # Set font for tick labels
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontfamily('Times New Roman')

        plt.tight_layout()
        plt.savefig(f'shap_force_plots{feature_suffix}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()
        print("Force plots created successfully!")

    except Exception as e:
        print(f"Force plots failed: {e}")
        print("Skipping force plots...")

    print(f"\nSHAP analysis completed successfully!")
    print(f"Files saved:")
    print(f"  - shap_feature_importance{feature_suffix}.csv (top non-temperature features)")
    print(f"  - shap_all_features_importance{feature_suffix}.csv (all non-temperature features)")
    print(f"  - shap_detailed_analysis{feature_suffix}.csv")
    print(f"  - Multiple high-quality PNG plots for journal submission")
    print(f"Note: Temperature-related features excluded from analysis and display")

    # Add categorized plot
    print("Creating professor's preferred categorized SHAP plot...")
    fig_categorized, ax_categorized = add_categorized_plot_to_existing_analysis(
       feature_importance_df[non_temp_mask], n_top_features, feature_suffix
    )

    return feature_importance_df, shap_values_top

def train_model_with_cv(file_path, n_features=None, cv_folds=5):
    """
    Train model with cross-validation and comprehensive analysis
    """

    # OPTIMIZED PARAMETERS FROM LATEST HYPERPARAMETER SEARCH
    best_params = {
        'hidden_layers': 2,
        'units': 480,
        'dropout_rate': 0.10069810416854146,
        'learning_rate': 0.002709138967161228,
        'batch_size': 16,
        'l2_lambda': 1.2535738166789118e-06
    }

    print(f"Training with {cv_folds}-Fold Cross-Validation")
    print("="*60)

    # Load and prepare data
    X_train_full, X_test, y_train_full, y_test, preprocessor, feature_names = load_and_prepare_data(file_path)

    # Apply feature selection if specified
    if n_features is not None and n_features < X_train_full.shape[1]:
        X_temp_train, X_temp_val, y_temp_train, y_temp_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        X_train_selected, X_temp_val_selected, X_test_selected, selected_features = select_features(
            X_temp_train, y_temp_train, X_temp_val, X_test, n_features
        )
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_train_full_selected = selector.fit_transform(X_train_full, y_train_full)
        X_train_full = pd.DataFrame(X_train_full_selected, columns=selected_features)
        X_test = X_test_selected
        feature_names = selected_features

    print(f"Dataset: {X_train_full.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train_full.shape[1]}")
    print("="*60)

    # Cross-validation setup
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = []
    all_fold_results = []
    all_training_curves = []

    print(f"Starting {cv_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_full)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        print("-" * 30)

        # Split data for this fold
        X_train_fold = X_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        y_val_fold = y_train_full.iloc[val_idx]

        # Create model for this fold
        model = ViscosityNN(
            input_dim=X_train_fold.shape[1],
            hidden_layers=best_params["hidden_layers"],
            units=best_params["units"],
            dropout_rate=best_params["dropout_rate"]
        )

        # Train model for this fold
        model, train_losses, val_losses = train_model(
            model, X_train_fold.values, y_train_fold, X_val_fold.values, y_val_fold,
            epochs=300,
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            l2_lambda=best_params["l2_lambda"],
            patience=30
        )

        # Store training curves
        all_training_curves.append({
            'fold': fold + 1,
            'train_losses': train_losses,
            'val_losses': val_losses
        })

        # Evaluate this fold
        train_results = evaluate_model(model, X_train_fold.values, y_train_fold)
        val_results = evaluate_model(model, X_val_fold.values, y_val_fold)
        test_results = evaluate_model(model, X_test.values, y_test)

        # Store detailed results for plotting
        all_fold_results.append({
            'fold': fold + 1,
            'model': model,
            'train_results': train_results,
            'val_results': val_results,
            'test_results': test_results
        })

        # Store CV results summary
        fold_results = {
            'fold': fold + 1,
            'train_aard': train_results['aard'],
            'train_r2': train_results['r2'],
            'train_rmse': train_results['rmse'],
            'train_mae': train_results['mae'],
            'val_aard': val_results['aard'],
            'val_r2': val_results['r2'],
            'val_rmse': val_results['rmse'],
            'val_mae': val_results['mae'],
            'test_aard': test_results['aard'],
            'test_r2': test_results['r2'],
            'test_rmse': test_results['rmse'],
            'test_mae': test_results['mae'],
            'train_max_ard': train_results['max_ard'],
            'val_max_ard': val_results['max_ard'],
            'test_max_ard': test_results['max_ard']
        }
        cv_results.append(fold_results)

        print(f"Fold {fold + 1} Results:")
        print(f"  Train: AARD={train_results['aard']:.2f}%, R²={train_results['r2']:.4f}")
        print(f"  Val:   AARD={val_results['aard']:.2f}%, R²={val_results['r2']:.4f}")
        print(f"  Test:  AARD={test_results['aard']:.2f}%, R²={test_results['r2']:.4f}")

    # Calculate CV statistics
    cv_df = pd.DataFrame(cv_results)

    # Create feature suffix for file naming
    feature_suffix = f"_{n_features}features" if n_features else "_allfeatures"

    # Print comprehensive CV results
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS SUMMARY ({cv_folds} FOLDS)")
    print(f"{'='*80}")

    # Calculate and display statistics
    metrics = ['train_aard', 'val_aard', 'test_aard', 'train_r2', 'val_r2', 'test_r2',
               'train_rmse', 'val_rmse', 'test_rmse', 'train_mae', 'val_mae', 'test_mae',
               'train_max_ard', 'val_max_ard', 'test_max_ard']

    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<8} {'Min':<8} {'Max':<8} {'95% CI':<15}")
    print("-" * 70)

    summary_stats = {}
    for metric in metrics:
        values = cv_df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
        ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))

        summary_stats[metric] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

        print(f"{metric:<15} {mean_val:<10.3f} {std_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    # Enhanced results table
    print(f"\n{'='*80}")
    print(f"DETAILED PERFORMANCE METRICS")
    print(f"{'='*80}")

    print(f"\nTraining Set Performance:")
    print(f"  AARD: {summary_stats['train_aard']['mean']:.2f} ± {summary_stats['train_aard']['std']:.2f}%")
    print(f"  R²:   {summary_stats['train_r2']['mean']:.4f} ± {summary_stats['train_r2']['std']:.4f}")
    print(f"  RMSE: {summary_stats['train_rmse']['mean']:.4f} ± {summary_stats['train_rmse']['std']:.4f}")
    print(f"  MAE:  {summary_stats['train_mae']['mean']:.4f} ± {summary_stats['train_mae']['std']:.4f}")
    print(f"  Max ARD: {summary_stats['train_max_ard']['mean']:.2f} ± {summary_stats['train_max_ard']['std']:.2f}%")

    print(f"\nValidation Set Performance:")
    print(f"  AARD: {summary_stats['val_aard']['mean']:.2f} ± {summary_stats['val_aard']['std']:.2f}%")
    print(f"  R²:   {summary_stats['val_r2']['mean']:.4f} ± {summary_stats['val_r2']['std']:.4f}")
    print(f"  RMSE: {summary_stats['val_rmse']['mean']:.4f} ± {summary_stats['val_rmse']['std']:.4f}")
    print(f"  MAE:  {summary_stats['val_mae']['mean']:.4f} ± {summary_stats['val_mae']['std']:.4f}")
    print(f"  Max ARD: {summary_stats['val_max_ard']['mean']:.2f} ± {summary_stats['val_max_ard']['std']:.2f}%")

    print(f"\nTest Set Performance:")
    print(f"  AARD: {summary_stats['test_aard']['mean']:.2f} ± {summary_stats['test_aard']['std']:.2f}%")
    print(f"  R²:   {summary_stats['test_r2']['mean']:.4f} ± {summary_stats['test_r2']['std']:.4f}")
    print(f"  RMSE: {summary_stats['test_rmse']['mean']:.4f} ± {summary_stats['test_rmse']['std']:.4f}")
    print(f"  MAE:  {summary_stats['test_mae']['mean']:.4f} ± {summary_stats['test_mae']['std']:.4f}")
    print(f"  Max ARD: {summary_stats['test_max_ard']['mean']:.2f} ± {summary_stats['test_max_ard']['std']:.2f}%")

    # Fold-by-fold results table
    print(f"\n{'='*80}")
    print(f"FOLD-BY-FOLD DETAILED RESULTS")
    print(f"{'='*80}")
    print(f"{'Fold':<5} {'Train_AARD':<12} {'Val_AARD':<10} {'Test_AARD':<10} {'Train_R²':<10} {'Val_R²':<8} {'Test_R²':<8}")
    print("-" * 70)
    for _, row in cv_df.iterrows():
        print(f"{int(row['fold']):<5} {row['train_aard']:<12.2f} {row['val_aard']:<10.2f} {row['test_aard']:<10.2f} "
              f"{row['train_r2']:<10.4f} {row['val_r2']:<8.4f} {row['test_r2']:<8.4f}")

    # Create all plots
    print(f"\n{'='*50}")
    print("GENERATING COMPREHENSIVE PLOTS")
    print(f"{'='*50}")

    # 1. CV Parity and Residual Plots
    create_cv_plots(all_fold_results, feature_suffix)

    # 2. Training Curves Plot
    plt.figure(figsize=(15, 7))

    # Plot individual fold curves
    for i, curve_data in enumerate(all_training_curves):
        plt.subplot(2, 3, i+1)
        plt.plot(curve_data['train_losses'], label='Training', alpha=0.8, color='blue')
        plt.plot(curve_data['val_losses'], label='Validation', alpha=0.8, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.title(f'Fold {curve_data["fold"]} Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Average training curves
    plt.subplot(2, 3, 6)
    all_train_curves = [curve['train_losses'] for curve in all_training_curves]
    all_val_curves = [curve['val_losses'] for curve in all_training_curves]

    # Pad curves to same length
    max_length = max(len(curve) for curve in all_train_curves)
    padded_train = [curve + [curve[-1]] * (max_length - len(curve)) for curve in all_train_curves]
    padded_val = [curve + [curve[-1]] * (max_length - len(curve)) for curve in all_val_curves]

    mean_train = np.mean(padded_train, axis=0)
    mean_val = np.mean(padded_val, axis=0)
    std_train = np.std(padded_train, axis=0)
    std_val = np.std(padded_val, axis=0)

    epochs = range(len(mean_train))
    plt.plot(epochs, mean_train, label='Training (mean)', color='blue', linewidth=2)
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2, color='blue')
    plt.plot(epochs, mean_val, label='Validation (mean)', color='green', linewidth=2)
    plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Average Training Progress (5-Fold CV)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'cv_training_curves{feature_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Performance Box Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # AARD Box Plot
    aard_data = [cv_df['train_aard'], cv_df['val_aard'], cv_df['test_aard']]
    axes[0].boxplot(aard_data, labels=['Training', 'Validation', 'Test'])
    axes[0].set_ylabel('AARD (%)', fontweight='bold')
    #axes[0].set_title('AARD Distribution Across Folds', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # R² Box Plot
    r2_data = [cv_df['train_r2'], cv_df['val_r2'], cv_df['test_r2']]
    axes[1].boxplot(r2_data, labels=['Training', 'Validation', 'Test'])
    axes[1].set_ylabel('R²', fontweight='bold')
    #axes[1].set_title('R² Distribution Across Folds', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # RMSE Box Plot
    rmse_data = [cv_df['train_rmse'], cv_df['val_rmse'], cv_df['test_rmse']]
    axes[2].boxplot(rmse_data, labels=['Training', 'Validation', 'Test'])
    axes[2].set_ylabel('RMSE', fontweight='bold')
    #axes[2].set_title('RMSE Distribution Across Folds', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'cv_performance_boxplots{feature_suffix}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Train final model on full data for SHAP analysis
    print(f"\n{'='*50}")
    print("TRAINING FINAL MODEL FOR SHAP ANALYSIS")
    print(f"{'='*50}")

    # Create final model
    final_model = ViscosityNN(
        input_dim=X_train_full.shape[1],
        hidden_layers=best_params["hidden_layers"],
        units=best_params["units"],
        dropout_rate=best_params["dropout_rate"]
    )

    # Train final model on all training data
    final_model, _, _ = train_model(
        final_model, X_train_full.values, y_train_full,
        X_test.values[:100], y_test.values[:100],  
        epochs=300,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        l2_lambda=best_params["l2_lambda"],
        patience=30
    )

    # SHAP Analysis
    print(f"\n{'='*50}")
    print("GENERATING SHAP ANALYSIS")
    print(f"{'='*50}")

    feature_importance, shap_values = create_shap_analysis(
        final_model, X_train_full.values, feature_names, n_top_features=30, feature_suffix=feature_suffix
    )

    # Save all results
    print(f"\n{'='*50}")
    print("SAVING RESULTS")
    print(f"{'='*50}")

    # Save CV results
    cv_df.to_csv(f'cv_detailed_results{feature_suffix}.csv', index=False)

    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(f'cv_summary_statistics{feature_suffix}.csv')

    # Save final model
    torch.save(final_model.state_dict(), f'final_cv_model{feature_suffix}.pth')

    # Create final summary report
    with open(f'cv_final_report{feature_suffix}.txt', 'w') as f:
        f.write("CROSS-VALIDATION FINAL REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {X_train_full.shape[0]} training samples, {X_test.shape[0]} test samples\n")
        f.write(f"Features: {X_train_full.shape[1]}\n")
        f.write(f"Cross-validation folds: {cv_folds}\n\n")

        f.write("FINAL PERFORMANCE METRICS (Mean ± Std):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test Set AARD: {summary_stats['test_aard']['mean']:.2f} ± {summary_stats['test_aard']['std']:.2f}%\n")
        f.write(f"Test Set R²: {summary_stats['test_r2']['mean']:.4f} ± {summary_stats['test_r2']['std']:.4f}\n")
        f.write(f"Test Set RMSE: {summary_stats['test_rmse']['mean']:.4f} ± {summary_stats['test_rmse']['std']:.4f}\n")
        f.write(f"Test Set MAE: {summary_stats['test_mae']['mean']:.4f} ± {summary_stats['test_mae']['std']:.4f}\n")
        f.write(f"Test Set Max ARD: {summary_stats['test_max_ard']['mean']:.2f} ± {summary_stats['test_max_ard']['std']:.2f}%\n")

    print(f"All results saved with suffix: {feature_suffix}")
    print(f"Files generated:")
    print(f"  - cv_detailed_results{feature_suffix}.csv")
    print(f"  - cv_summary_statistics{feature_suffix}.csv")
    print(f"  - cv_final_report{feature_suffix}.txt")
    print(f"  - final_cv_model{feature_suffix}.pth")
    print(f"  - shap_feature_importance{feature_suffix}.csv")
    print(f"  - Multiple PNG plot files")

    return cv_df, summary_stats, feature_importance, final_model

# Main execution
if __name__ == '__main__':
    file_path = '/kaggle/input/Viscosity_Prediction_Project/Final_Processed_Data.csv'

    print("STARTING COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("=" * 80)

   
    cv_results, summary_stats, feature_importance, final_model = train_model_with_cv(
        file_path, n_features=None, cv_folds=5
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")

    # Print final summary
    print(f"\nFINAL SUMMARY:")
    print(f"Test Set Performance: {summary_stats['test_aard']['mean']:.2f} ± {summary_stats['test_aard']['std']:.2f}% AARD")
    print(f"Test Set R²: {summary_stats['test_r2']['mean']:.4f} ± {summary_stats['test_r2']['std']:.4f}")
    print(f"All plots and results have been saved for journal publication.")

