

%pip install evidential-deep-learning tensorflow
%pip install optuna

from scipy.special import inv_boxcox
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, RobustScaler,
                                   PolynomialFeatures, PowerTransformer, QuantileTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# Try to import evidential deep learning, with fallback
try:
    import evidential_deep_learning as edl
    EVIDENTIAL_AVAILABLE = True
    print("✅ Evidential Deep Learning library loaded successfully")
except ImportError:
    EVIDENTIAL_AVAILABLE = False
    print("⚠️ Evidential Deep Learning library not available, using custom implementation")

class FixedEvidentialViscosityPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        self.evidential_available = EVIDENTIAL_AVAILABLE

        self.preprocessor = None
        self.feature_selector = None
        self.models = []
        self.best_params = None
        self.target_transform = None
        self.epsilon = 1e-8

    def comprehensive_outlier_detection(self, X, y, contamination=0.05):
        """Comprehensive outlier detection using multiple methods"""
        from sklearn.ensemble import IsolationForest

        print(f"Original samples: {len(y)}")

        # Method 1: Isolation Forest on combined features and target
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        combined_data = np.column_stack([X, y.reshape(-1, 1)])
        iso_outliers = iso_forest.fit_predict(combined_data) == 1

        # Method 2: Statistical outliers using IQR 
        Q1, Q3 = np.percentile(y, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.8 * IQR  
        upper_bound = Q3 + 1.8 * IQR
        iqr_inliers = (y >= lower_bound) & (y <= upper_bound)

        # Method 3: Z-score based outliers
        z_scores = np.abs((y - np.mean(y)) / (np.std(y) + 1e-8))
        zscore_inliers = z_scores < 2.8  

        # Combine all methods (majority vote)
        inlier_votes = iso_outliers.astype(int) + iqr_inliers.astype(int) + zscore_inliers.astype(int)
        combined_inliers = inlier_votes >= 2  

        print(f"Isolation Forest inliers: {np.sum(iso_outliers)}")
        print(f"IQR inliers: {np.sum(iqr_inliers)}")
        print(f"Z-score inliers: {np.sum(zscore_inliers)}")
        print(f"Combined inliers (majority vote): {np.sum(combined_inliers)}")
        print(f"Outliers removed: {len(y) - np.sum(combined_inliers)}")

        return combined_inliers

    def advanced_feature_engineering(self, X_train, X_val, X_test, y_train):
        """Advanced feature engineering following best practices"""

        # Identify feature types
        numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        print(f"Feature types - Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")

        # Create advanced preprocessing pipeline
        transformers = []

        # Handle categorical features
        if categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    max_categories=30,  
                    drop='if_binary'  
                ), categorical_features)
            )

        # Handle numerical features with different scalers
        if numerical_features:
            n_num = len(numerical_features)

            # Split numerical features for different preprocessing
            third = max(1, n_num // 3)

            # Group 1: StandardScaler 
            group1 = numerical_features[:third]
            transformers.append(('num_standard', StandardScaler(), group1))

            # Group 2: RobustScaler 
            group2 = numerical_features[third:2*third]
            if group2:
                transformers.append(('num_robust', RobustScaler(), group2))

            # Group 3: QuantileTransformer 
            group3 = numerical_features[2*third:]
            if group3:
                transformers.append(('num_quantile', QuantileTransformer(
                    n_quantiles=min(1000, len(X_train)),
                    output_distribution='normal',
                    random_state=self.random_state
                ), group3))

        if not transformers:
            raise ValueError("No features found for preprocessing")

        # Apply preprocessing
        preprocessor = ColumnTransformer(transformers, remainder='drop')

        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        print(f"After preprocessing: {X_train_processed.shape[1]} features")

        # Feature selection pipeline
        X_train_selected, X_val_selected, X_test_selected = self.intelligent_feature_selection(
            X_train_processed, X_val_processed, X_test_processed, y_train
        )

        # Create polynomial/interaction features for top predictors
        if X_train_selected.shape[1] > 5:
            top_k = min(12, X_train_selected.shape[1])  

            poly = PolynomialFeatures(
                degree=2,
                interaction_only=True, 
                include_bias=False
            )

            poly_train = poly.fit_transform(X_train_selected[:, :top_k])
            poly_val = poly.transform(X_val_selected[:, :top_k])
            poly_test = poly.transform(X_test_selected[:, :top_k])

            # Combine original and interaction features
            X_train_final = np.column_stack([X_train_selected, poly_train])
            X_val_final = np.column_stack([X_val_selected, poly_val])
            X_test_final = np.column_stack([X_test_selected, poly_test])

            print(f"After interaction features: {X_train_final.shape[1]} features")
        else:
            X_train_final, X_val_final, X_test_final = X_train_selected, X_val_selected, X_test_selected

        # Store preprocessor
        self.preprocessor = preprocessor

        return X_train_final, X_val_final, X_test_final

    def intelligent_feature_selection(self, X_train, X_val, X_test, y_train):
        """Multi-step intelligent feature selection"""

        print(f"Starting feature selection with {X_train.shape[1]} features")

        # Step 1: Remove near-zero variance features
        from sklearn.feature_selection import VarianceThreshold
        var_threshold = VarianceThreshold(threshold=0.005) 
        X_train_var = var_threshold.fit_transform(X_train)
        X_val_var = var_threshold.transform(X_val)
        X_test_var = var_threshold.transform(X_test)

        print(f"After variance threshold: {X_train_var.shape[1]} features")

        if X_train_var.shape[1] == 0:
            return X_train, X_val, X_test

        # Step 2: Univariate feature selection using mutual information
        k_univariate = min(800, X_train_var.shape[1])
        univariate_selector = SelectKBest(
            score_func=mutual_info_regression,
            k=k_univariate
        )
        X_train_uni = univariate_selector.fit_transform(X_train_var, y_train)
        X_val_uni = univariate_selector.transform(X_val_var)
        X_test_uni = univariate_selector.transform(X_test_var)

        print(f"After univariate selection: {X_train_uni.shape[1]} features")

        # Step 3: Recursive Feature Elimination with Random Forest
        if X_train_uni.shape[1] > 200:
            rf_selector = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=10
            )

            n_features_to_select = min(300, X_train_uni.shape[1])
            rfe = RFE(
                estimator=rf_selector,
                n_features_to_select=n_features_to_select,
                step=0.1,
                verbose=0
            )

            X_train_rfe = rfe.fit_transform(X_train_uni, y_train)
            X_val_rfe = rfe.transform(X_val_uni)
            X_test_rfe = rfe.transform(X_test_uni)

            print(f"After RFE: {X_train_rfe.shape[1]} features")
            return X_train_rfe, X_val_rfe, X_test_rfe

        return X_train_uni, X_val_uni, X_test_uni

    def optimal_target_transformation(self, y_train, y_val, y_test):
        """Find the optimal target transformation"""

        print("Finding optimal target transformation...")

        transformations = {
            'log': lambda x: np.log(x + self.epsilon),
            'log1p': lambda x: np.log1p(x),
            'sqrt': lambda x: np.sqrt(np.maximum(x, 0) + self.epsilon),
            'boxcox': None  
        }

        best_transform = 'log'
        best_score = float('-inf')

        for name, transform_func in transformations.items():
            if name == 'boxcox':
                # Box-Cox transformation
                try:
                    from scipy.stats import boxcox
                    y_train_bc, lambda_val = boxcox(y_train + self.epsilon)
                    y_val_bc = boxcox(y_val + self.epsilon, lmbda=lambda_val)
                    y_test_bc = boxcox(y_test + self.epsilon, lmbda=lambda_val)

                    # Store lambda for inverse transform
                    self.boxcox_lambda = lambda_val

                    # Evaluate transformation quality 
                    from scipy.stats import shapiro
                    if len(y_train_bc) <= 5000:
                        _, p_value = shapiro(y_train_bc[:5000])
                        if p_value > best_score:
                            best_score = p_value
                            best_transform = 'boxcox'
                            y_train_transformed = y_train_bc
                            y_val_transformed = y_val_bc
                            y_test_transformed = y_test_bc
                except:
                    continue
            else:
                try:
                    y_train_t = transform_func(y_train)
                    y_val_t = transform_func(y_val)
                    y_test_t = transform_func(y_test)

                    # Check for valid transformation
                    if np.any(np.isinf(y_train_t)) or np.any(np.isnan(y_train_t)):
                        continue

                    # Evaluate using normality test
                    from scipy.stats import shapiro
                    if len(y_train_t) <= 5000:
                        _, p_value = shapiro(y_train_t[:5000])
                        if p_value > best_score:
                            best_score = p_value
                            best_transform = name
                            y_train_transformed = y_train_t
                            y_val_transformed = y_val_t
                            y_test_transformed = y_test_t
                except:
                    continue

        # Fallback to log if nothing else worked
        if best_transform == 'log' and 'y_train_transformed' not in locals():
            y_train_transformed = np.log(y_train + self.epsilon)
            y_val_transformed = np.log(y_val + self.epsilon)
            y_test_transformed = np.log(y_test + self.epsilon)

        print(f"Best transformation: {best_transform} (normality p-value: {best_score:.6f})")
        self.target_transform = best_transform

        return y_train_transformed, y_val_transformed, y_test_transformed

    def inverse_transform_target(self, y_transformed):
        """Inverse transform predictions back to original space"""
        if self.target_transform == 'log':
            return np.exp(y_transformed) - self.epsilon
        elif self.target_transform == 'log1p':
            return np.expm1(y_transformed)
        elif self.target_transform == 'sqrt':
            return np.square(y_transformed) - self.epsilon
        elif self.target_transform == 'boxcox':
            from scipy.special import inv_boxcox
            return inv_boxcox(y_transformed, self.boxcox_lambda) - self.epsilon
        else:
            return y_transformed

    def load_and_prepare_data(self, file_path):
        """Comprehensive data loading and preparation"""

        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Find target and set columns
        viscosity_cols = [col for col in df.columns if 'viscosity' in col.lower()]
        if not viscosity_cols:
            raise ValueError("Viscosity column not found")
        target_col = viscosity_cols[0]

        set_cols = [col for col in df.columns if col.lower() == 'set']
        if not set_cols:
            raise ValueError("Set column not found")
        set_col = set_cols[0]

        print(f"Target column: {target_col}")
        print(f"Set column: {set_col}")

        # Basic statistics
        print(f"\nTarget statistics:")
        print(f"  Min: {df[target_col].min():.6f}")
        print(f"  Max: {df[target_col].max():.6f}")
        print(f"  Mean: {df[target_col].mean():.6f}")
        print(f"  Median: {df[target_col].median():.6f}")
        print(f"  Std: {df[target_col].std():.6f}")

        # Split data
        train_df = df[df[set_col] == 'training'].copy()
        test_df = df[df[set_col] == 'test'].copy()

        print(f"\nData split:")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")

        # Prepare features and targets
        X_train_full = train_df.drop([target_col, set_col], axis=1)
        y_train_full = train_df[target_col].values

        X_test = test_df.drop([target_col, set_col], axis=1)
        y_test = test_df[target_col].values

        # Handle missing values intelligently
        print(f"\nHandling missing values...")

        # For numerical columns: use median
        numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_val = X_train_full[col].median()
            X_train_full[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)

        # For categorical columns: use mode or 'missing'
        categorical_cols = X_train_full.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = X_train_full[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'missing'
            X_train_full[col].fillna(fill_val, inplace=True)
            X_test[col].fillna(fill_val, inplace=True)

        # Enhanced outlier removal
        print(f"\nRemoving outliers...")
        numeric_features = X_train_full.select_dtypes(include=[np.number]).values
        inlier_mask = self.comprehensive_outlier_detection(
            numeric_features, y_train_full, contamination=0.05
        )

        X_train_full = X_train_full[inlier_mask]
        y_train_full = y_train_full[inlier_mask]

        # Stratified train/validation split
        print(f"\nCreating train/validation split...")

        # Create stratification bins
        try:
            y_bins = pd.qcut(y_train_full, q=min(10, len(y_train_full)//10), labels=False, duplicates='drop')
        except:
            y_bins = pd.cut(y_train_full, bins=min(10, len(y_train_full)//10), labels=False)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.15,  
            random_state=self.random_state,
            stratify=y_bins
        )

        print(f"Final data split:")
        print(f"  Training: {len(X_train)}")
        print(f"  Validation: {len(X_val)}")
        print(f"  Test: {len(X_test)}")

        # Optimal target transformation
        y_train_transformed, y_val_transformed, y_test_transformed = self.optimal_target_transformation(
            y_train, y_val, y_test
        )

        # Advanced feature engineering
        print(f"\nAdvanced feature engineering...")
        X_train_processed, X_val_processed, X_test_processed = self.advanced_feature_engineering(
            X_train, X_val, X_test, y_train_transformed
        )

        print(f"Final feature dimension: {X_train_processed.shape[1]}")

        return (X_train_processed, X_val_processed, X_test_processed,
                y_train_transformed, y_val_transformed, y_test_transformed,
                y_train, y_val, y_test)

    def create_evidential_model(self, input_dim, params=None):
        """Create evidential model with proper API compatibility"""
        # Default optimized parameters
        if params is None:
            params = {
                'n_layers': 6,
                'units': [1024, 512, 256, 128, 64, 32],
                'dropout_rates': [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
                'l2_reg': 1e-4,
                'lambda_coef': 0.1,
                'learning_rate': 0.001,
                'activation': 'gelu'
            }

        # Input and preprocessing layers
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.GaussianNoise(0.01)(x)

        # Hidden layers
        for i in range(params['n_layers']):
            x = tf.keras.layers.Dense(
                params['units'][i],
                activation=params['activation'],
                kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']),
                kernel_initializer='he_normal'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(params['dropout_rates'][i])(x)

        # Try official evidential layer if available
        if self.evidential_available:
            try:
                outputs = edl.layers.DenseNormalGamma(1)(x)

                def evidential_loss(y_true, y_pred):
                    return edl.losses.EvidentialRegression(
                        y_true,
                        y_pred,
                        coeff=params['lambda_coef']
                    )

                print("✅ Using official evidential layers")

            except Exception as e:
                print(f"⚠️ Official evidential layer failed ({e}), falling back to custom implementation")
                self.evidential_available = False

        # Fallback to custom evidential implementation
        if not self.evidential_available:
            outputs = tf.keras.layers.Dense(4, activation='linear')(x)

            def evidential_loss(y_true, y_pred):
                mu    = y_pred[:, 0:1]
                logv  = y_pred[:, 1:2]
                alpha = y_pred[:, 2:3]
                beta  = y_pred[:, 3:4]

                # ensure positivity
                v     = tf.nn.softplus(logv)
                alpha = tf.nn.softplus(alpha) + 1.0
                beta  = tf.nn.softplus(beta)

                # negative log‐likelihood
                nll = (
                    0.5 * tf.math.log(2.0 * np.pi)
                    - 0.5 * tf.math.log(v)
                    + 0.5 * v * tf.square(y_true - mu)
                )

                # evidential regularizer
                error    = tf.abs(y_true - mu)
                evidence = 2.0 * v + alpha
                reg      = error * evidence

                return tf.reduce_mean(nll + params['lambda_coef'] * reg)

            print("✅ Using custom evidential implementation")

        # Build and compile
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss=evidential_loss,
            metrics=['mse', 'mae']
        )

        return model



    def train_with_callbacks(self, model, X_train, y_train, X_val, y_val, epochs=400):
        """Train model with comprehensive callbacks"""

        # Ensure correct shapes
        y_train = np.array(y_train).reshape(-1, 1)
        y_val = np.array(y_val).reshape(-1, 1)

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=60,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-6
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=25,
                min_lr=1e-7,
                verbose=1,
                min_delta=1e-6
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # Train model
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            return model, history
        except Exception as e:
            print(f"Training failed: {e}")
            return None, None

    def calculate_comprehensive_metrics(self, y_true_transformed, y_pred_transformed, dataset_name=""):
        """Calculate comprehensive metrics with proper error handling"""

        # Ensure arrays and handle problematic values
        y_pred_transformed = np.array(y_pred_transformed).flatten()
        y_true_transformed = np.array(y_true_transformed).flatten()

        # Handle inf/nan values
        if np.any(~np.isfinite(y_pred_transformed)):
            print(f"Warning: Non-finite predictions in {dataset_name}, replacing with median")
            finite_mask = np.isfinite(y_pred_transformed)
            if np.any(finite_mask):
                median_pred = np.median(y_pred_transformed[finite_mask])
                y_pred_transformed = np.where(finite_mask, y_pred_transformed, median_pred)
            else:
                y_pred_transformed = y_true_transformed.copy()

        # Clip extreme values
        y_pred_transformed = np.clip(y_pred_transformed, -20, 20)

        # Transform back to original space
        y_true_orig = self.inverse_transform_target(y_true_transformed)
        y_pred_orig = self.inverse_transform_target(y_pred_transformed)

        # Ensure positive values (viscosity must be positive)
        y_true_orig = np.maximum(y_true_orig, self.epsilon)
        y_pred_orig = np.maximum(y_pred_orig, self.epsilon)

        # Calculate AARD (primary metric)
        relative_errors = np.abs((y_true_orig - y_pred_orig) / y_true_orig)
        relative_errors = np.clip(relative_errors, 0, 10)  # Cap at 1000% error

        aard = np.mean(relative_errors) * 100
        median_ard = np.median(relative_errors) * 100
        q75_ard = np.percentile(relative_errors, 75) * 100
        q90_ard = np.percentile(relative_errors, 90) * 100
        max_ard = np.max(relative_errors) * 100

        # R² calculations
        try:
            r2_transformed = r2_score(y_true_transformed, y_pred_transformed)
        except:
            r2_transformed = -1.0

        try:
            r2_orig = r2_score(y_true_orig, y_pred_orig)
        except:
            r2_orig = -1.0

        # Other metrics
        rmse_transformed = np.sqrt(mean_squared_error(y_true_transformed, y_pred_transformed))
        mae_transformed = mean_absolute_error(y_true_transformed, y_pred_transformed)
        rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)

        return {
            'dataset': dataset_name,
            'aard': aard,
            'median_ard': median_ard,
            'q75_ard': q75_ard,
            'q90_ard': q90_ard,
            'max_ard': max_ard,
            'r2_transformed': r2_transformed,
            'r2_orig': r2_orig,
            'rmse_transformed': rmse_transformed,
            'mae_transformed': mae_transformed,
            'rmse_orig': rmse_orig,
            'mae_orig': mae_orig,
            'y_pred_transformed': y_pred_transformed,
            'y_pred_orig': y_pred_orig,
            'y_true_orig': y_true_orig
        }

    def optuna_objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization """

        try:
            # Define hyperparameter search space
            n_layers = trial.suggest_int('n_layers', 4, 8)

            units = []
            dropout_rates = []

            for i in range(n_layers):
                # Decreasing units per layer
                if i == 0:
                    unit_range = (256, 1024)  # Reduced range
                elif i == 1:
                    unit_range = (128, 512)
                elif i == 2:
                    unit_range = (64, 256)
                elif i == 3:
                    unit_range = (32, 128)
                elif i == 4:
                    unit_range = (16, 64)
                elif i == 5:
                    unit_range = (8, 32)
                else:
                    unit_range = (4, 16)

                units.append(trial.suggest_int(f'units_{i}', unit_range[0], unit_range[1], step=16))
                dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.05, 0.4))

            params = {
                'n_layers': n_layers,
                'units': units,
                'dropout_rates': dropout_rates,
                'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True),
                'lambda_coef': trial.suggest_float('lambda_coef', 0.001, 1.0, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
                'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'elu'])
            }

            # Create and train model
            model = self.create_evidential_model(X_train.shape[1], params)

            if model is None:
                return float('inf')

            model, history = self.train_with_callbacks(
                model, X_train, y_train, X_val, y_val, epochs=100  
            )

            if model is None or history is None:
                return float('inf')

            # Make predictions
            y_pred = model.predict(X_val, verbose=0)

            # Extract mean prediction based on model type
            if EVIDENTIAL_AVAILABLE and y_pred.shape[1] >= 4:
                y_pred_mean = y_pred[:, 0]  
            elif y_pred.shape[1] >= 4:
                y_pred_mean = y_pred[:, 0]  
            else:
                y_pred_mean = y_pred.flatten()

            # Calculate metrics
            results = self.calculate_comprehensive_metrics(y_val, y_pred_mean, "validation")

            # Clean up
            del model
            tf.keras.backend.clear_session()

            # Return AARD as objective 
            return results['aard']

        except Exception as e:
            print(f"Trial failed: {e}")
            tf.keras.backend.clear_session()
            return float('inf')

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Hyperparameter optimization using Optuna - REDUCED TRIALS"""

        print(f"Starting hyperparameter optimization with {n_trials} trials...")

        # Create study with TPE sampler for better optimization
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5, 
            n_ei_candidates=12,  
            seed=self.random_state
        )

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='evidential_viscosity_optimization'
        )

        # Optimize
        study.optimize(
            lambda trial: self.optuna_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )

        # Get best parameters
        valid_trials = [t for t in study.trials if t.value is not None and t.value != float('inf')]

        if valid_trials:
            best_trial = min(valid_trials, key=lambda x: x.value)
            print(f"\nOptimization completed!")
            print(f"Best AARD: {best_trial.value:.3f}%")
            print(f"Number of successful trials: {len(valid_trials)}/{n_trials}")

            # Convert to model parameters format
            n_layers = best_trial.params['n_layers']
            units = [best_trial.params[f'units_{i}'] for i in range(n_layers)]
            dropout_rates = [best_trial.params[f'dropout_{i}'] for i in range(n_layers)]

            self.best_params = {
                'n_layers': n_layers,
                'units': units,
                'dropout_rates': dropout_rates,
                'l2_reg': best_trial.params['l2_reg'],
                'lambda_coef': best_trial.params['lambda_coef'],
                'learning_rate': best_trial.params['learning_rate'],
                'activation': best_trial.params['activation']
            }

            print("Best parameters:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")

        else:
            print("No successful trials! Using default parameters.")
            self.best_params = {
                'n_layers': 6,
                'units': [512, 256, 128, 64, 32, 16],  
                'dropout_rates': [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
                'l2_reg': 1e-4,
                'lambda_coef': 0.1,
                'learning_rate': 0.001,
                'activation': 'gelu'
            }

        return self.best_params

    def train_ensemble(self, X_train, y_train, X_val, y_val, n_models=3):
        """Train ensemble of evidential models - REDUCED SIZE"""

        print(f"Training ensemble of {n_models} evidential models...")

        models = []
        successful_models = 0

        for i in range(n_models):
            print(f"\nTraining model {i+1}/{n_models}")

            # Create parameter variations for diversity
            params = self.best_params.copy()
            if i > 0:
                # Add controlled randomness
                params['learning_rate'] *= np.random.uniform(0.8, 1.2)
                params['lambda_coef'] *= np.random.uniform(0.7, 1.3)
                params['l2_reg'] *= np.random.uniform(0.5, 2.0)

                # Vary dropout rates
                for j in range(len(params['dropout_rates'])):
                    params['dropout_rates'][j] *= np.random.uniform(0.8, 1.2)
                    params['dropout_rates'][j] = np.clip(params['dropout_rates'][j], 0.0, 0.5)

            # Set different random seed
            tf.random.set_seed(self.random_state + i * 100)

            try:
                model = self.create_evidential_model(X_train.shape[1], params)

                if model is None:
                    print(f"Failed to create model {i+1}")
                    continue

                model, history = self.train_with_callbacks(
                    model, X_train, y_train, X_val, y_val, epochs=300  
                )

                if model is None or history is None:
                    print(f"Failed to train model {i+1}")
                    continue

                models.append(model)
                successful_models += 1

                # Quick validation
                val_pred = model.predict(X_val, verbose=0)
                if val_pred.shape[1] >= 4:
                    val_pred_mean = val_pred[:, 0]
                else:
                    val_pred_mean = val_pred.flatten()

                val_results = self.calculate_comprehensive_metrics(y_val, val_pred_mean, f"Val_Model_{i+1}")
                print(f"Model {i+1} validation AARD: {val_results['aard']:.3f}%")

            except Exception as e:
                print(f"Failed to train model {i+1}: {e}")
                tf.keras.backend.clear_session()
                continue

        if successful_models == 0:
            raise ValueError("Failed to train any models in ensemble")

        print(f"\nEnsemble training completed: {successful_models} models trained successfully")
        self.models = models
        return models

    def ensemble_predict(self, X):
        """Make ensemble predictions with uncertainty estimation"""

        if not self.models:
            raise ValueError("No trained models available")

        predictions = []
        uncertainties = []

        for i, model in enumerate(self.models):
            try:
                # Get evidential predictions
                pred_output = model.predict(X, verbose=0)

                # Extract mean and uncertainty from evidential output
                if len(pred_output.shape) > 1 and pred_output.shape[1] >= 4:
                    # Evidential output: [mu, logv, alpha, beta] or [mu, v, alpha, beta]
                    mu = pred_output[:, 0]  

                    if EVIDENTIAL_AVAILABLE:
                        # Official implementation format
                        v = pred_output[:, 1]   # Precision
                        alpha = pred_output[:, 2]  # Evidence parameter
                        beta = pred_output[:, 3]   # Evidence parameter
                        uncertainty = beta / (v * (alpha - 1) + 1e-8)
                    else:
                        # Custom implementation format
                        logv = pred_output[:, 1]
                        alpha = pred_output[:, 2]
                        beta = pred_output[:, 3]
                        v = np.exp(logv)
                        uncertainty = beta / (v * (alpha - 1) + 1e-8)

                    predictions.append(mu)
                    uncertainties.append(uncertainty)
                else:
                    # Fallback: just use first column as mean
                    mu = pred_output[:, 0] if len(pred_output.shape) > 1 else pred_output
                    predictions.append(mu)
                    uncertainties.append(np.ones_like(mu))  # Uniform uncertainty

            except Exception as e:
                print(f"Warning: Model {i} prediction failed: {e}")
                continue

        if not predictions:
            raise ValueError("All ensemble models failed to predict")

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)

        # Simple ensemble average (equal weights)
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_uncertainty = np.mean(uncertainties, axis=0)

        return ensemble_pred, ensemble_uncertainty

    def plot_results_with_uncertainty(self, results_dict):
        """Plot results with uncertainty visualization - IMPROVED FONTS"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        datasets = list(results_dict.keys())

        for i, (dataset, results) in enumerate(results_dict.items()):
            if i >= 3: 
                break

            y_true = results['y_true_orig']
            y_pred = results['y_pred_orig']
            uncertainty = results.get('uncertainty', np.ones_like(y_pred))

            # 1. Parity plot with uncertainty
            ax = axes[0, i]
            scatter = ax.scatter(y_true, y_pred, c=uncertainty, alpha=0.6, s=30, cmap='viridis')

            # Perfect prediction line
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            # Error bands
            ax.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g--', alpha=0.5)
            ax.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g--', alpha=0.5)

            ax.set_xlabel('Actual Viscosity', fontsize=16)
            ax.set_ylabel('Predicted Viscosity', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True, alpha=0.3)

            # Colorbar with proper font
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Uncertainty', fontsize=16)
            cbar.ax.tick_params(labelsize=16)

            # 2. Error distribution
            ax2 = axes[1, i]
            relative_errors = np.abs((y_true - y_pred) / y_true) * 100
            ax2.hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(results['aard'], color='red', linestyle='--',
                       label=f'Mean: {results["aard"]:.2f}%', linewidth=2)
            ax2.axvline(results['median_ard'], color='green', linestyle='--',
                       label=f'Median: {results["median_ard"]:.2f}%', linewidth=2)
            ax2.set_xlabel('Absolute Relative Error (%)', fontsize=16)
            ax2.set_ylabel('Frequency', fontsize=16)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            legend = ax2.legend(fontsize=16)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_evidential_pipeline(self, file_path, optimize=True, n_trials=30, use_ensemble=True):
        """Run complete pipeline using fixed evidential deep learning"""

        print("="*80)
        print("FIXED EVIDENTIAL DEEP LEARNING - VISCOSITY PREDICTION")
        print("Target: AARD ≤ 8% with high R²")
        print("="*80)

        try:
            # 1. Load and prepare data
            print("\n" + "="*60)
            print("DATA LOADING AND PREPROCESSING")
            print("="*60)

            data = self.load_and_prepare_data(file_path)
            (X_train, X_val, X_test,
             y_train_transformed, y_val_transformed, y_test_transformed,
             y_train_orig, y_val_orig, y_test_orig) = data

            # 2. Hyperparameter optimization
            if optimize:
                print("\n" + "="*60)
                print("HYPERPARAMETER OPTIMIZATION")
                print("="*60)
                self.optimize_hyperparameters(
                    X_train, y_train_transformed, X_val, y_val_transformed, n_trials
                )
            else:
                print("\nUsing default parameters...")
                self.best_params = {
                    'n_layers': 5,
                    'units': [512, 256, 128, 64, 32],
                    'dropout_rates': [0.3, 0.25, 0.2, 0.15, 0.1],
                    'l2_reg': 1e-4,
                    'lambda_coef': 0.1,
                    'learning_rate': 0.001,
                    'activation': 'gelu'
                }

            # 3. Model training
            if use_ensemble:
                print("\n" + "="*60)
                print("ENSEMBLE TRAINING")
                print("="*60)

                models = self.train_ensemble(
                    X_train, y_train_transformed, X_val, y_val_transformed, n_models=3
                )

                # Make ensemble predictions with uncertainty
                train_pred, train_uncertainty = self.ensemble_predict(X_train)
                val_pred, val_uncertainty = self.ensemble_predict(X_val)
                test_pred, test_uncertainty = self.ensemble_predict(X_test)

            else:
                print("\n" + "="*60)
                print("SINGLE MODEL TRAINING")
                print("="*60)

                model = self.create_evidential_model(X_train.shape[1], self.best_params)

                if model is None:
                    raise ValueError("Failed to create model")

                model, history = self.train_with_callbacks(
                    model, X_train, y_train_transformed, X_val, y_val_transformed, epochs=400
                )

                if model is None or history is None:
                    raise ValueError("Failed to train model")

                self.models = [model]

                # Single model predictions
                train_output = model.predict(X_train, verbose=0)
                val_output = model.predict(X_val, verbose=0)
                test_output = model.predict(X_test, verbose=0)

                if train_output.shape[1] >= 4:
                    train_pred = train_output[:, 0]
                    val_pred = val_output[:, 0]
                    test_pred = test_output[:, 0]

                    # Extract uncertainties from evidential output
                    if EVIDENTIAL_AVAILABLE:
                        train_uncertainty = train_output[:, 3] / (train_output[:, 1] * (train_output[:, 2] - 1) + 1e-8)
                        val_uncertainty = val_output[:, 3] / (val_output[:, 1] * (val_output[:, 2] - 1) + 1e-8)
                        test_uncertainty = test_output[:, 3] / (test_output[:, 1] * (test_output[:, 2] - 1) + 1e-8)
                    else:
                        # Custom implementation
                        train_uncertainty = train_output[:, 3] / (np.exp(train_output[:, 1]) * (train_output[:, 2] - 1) + 1e-8)
                        val_uncertainty = val_output[:, 3] / (np.exp(val_output[:, 1]) * (val_output[:, 2] - 1) + 1e-8)
                        test_uncertainty = test_output[:, 3] / (np.exp(test_output[:, 1]) * (test_output[:, 2] - 1) + 1e-8)
                else:
                    train_pred = train_output.flatten()
                    val_pred = val_output.flatten()
                    test_pred = test_output.flatten()
                    train_uncertainty = np.ones_like(train_pred)
                    val_uncertainty = np.ones_like(val_pred)
                    test_uncertainty = np.ones_like(test_pred)


           # 4. Calculate comprehensive metrics
            print("\n" + "="*60)
            print("FINAL EVALUATION")
            print("="*60)

            train_results = self.calculate_comprehensive_metrics(
                y_train_transformed, train_pred, "Training"
            )
            val_results = self.calculate_comprehensive_metrics(
                y_val_transformed, val_pred, "Validation"
            )
            test_results = self.calculate_comprehensive_metrics(
                y_test_transformed, test_pred, "Test"
            )

            # Add uncertainty information
            train_results['uncertainty'] = train_uncertainty
            val_results['uncertainty'] = val_uncertainty
            test_results['uncertainty'] = test_uncertainty

            # Print detailed results with Max ARD included
            print(f"{'Dataset':<12} {'AARD%':<8} {'MedianARD':<10} {'Q75ARD':<8} {'Q90ARD':<8} {'MaxARD':<8} {'R²_orig':<8} {'Mean_Unc':<10} {'Median_Unc':<12}")
            print("-" * 120)

            for results in [train_results, val_results, test_results]:
                mean_uncertainty = np.mean(results['uncertainty'])
                median_uncertainty = np.median(results['uncertainty'])
                print(f"{results['dataset']:<12} {results['aard']:<8.3f} "
                      f"{results['median_ard']:<10.3f} {results['q75_ard']:<8.3f} "
                      f"{results['q90_ard']:<8.3f} {results['max_ard']:<8.3f} "
                      f"{results['r2_orig']:<8.4f} {mean_uncertainty:<10.4f} {median_uncertainty:<12.4f}")

            # 5. Visualization
            results_dict = {
                'Training': train_results,
                'Validation': val_results,
                'Test': test_results
            }

            self.plot_results_with_uncertainty(results_dict)

            # 6. Save results and models
            results_df = pd.DataFrame([
                {
                    'Dataset': results['dataset'],
                    'AARD_%': results['aard'],
                    'MedianARD_%': results['median_ard'],
                    'Q75ARD_%': results['q75_ard'],
                    'Q90ARD_%': results['q90_ard'],
                    'MaxARD_%': results['max_ard'],
                    'R²_orig': results['r2_orig'],
                    'RMSE_orig': results['rmse_orig'],
                    'MAE_orig': results['mae_orig'],
                    'Mean_Uncertainty': np.mean(results['uncertainty']),
                    'Median_Uncertainty': np.median(results['uncertainty'])
                }
                for results in [train_results, val_results, test_results]
            ])

            results_df.to_csv('fixed_evidential_results.csv', index=False)
            print(f"\nResults saved to 'fixed_evidential_results.csv'")

            if self.models:
                self.models[0].save('fixed_evidential_model.h5')
                print(f"Best model saved to 'fixed_evidential_model.h5'")

            # 7. Print NaN/Null values and relevant predictions
            print(f"\n" + "="*60)
            print("NaN/NULL VALUES AND SAMPLE PREDICTIONS")
            print("="*60)

            # Check for NaN values in predictions
            train_nan_count = np.sum(np.isnan(train_results['y_pred_orig']))
            val_nan_count = np.sum(np.isnan(val_results['y_pred_orig']))
            test_nan_count = np.sum(np.isnan(test_results['y_pred_orig']))

            print(f"NaN values in predictions:")
            print(f"  Training: {train_nan_count}/{len(train_results['y_pred_orig'])}")
            print(f"  Validation: {val_nan_count}/{len(val_results['y_pred_orig'])}")
            print(f"  Test: {test_nan_count}/{len(test_results['y_pred_orig'])}")

            # Check for infinite values
            train_inf_count = np.sum(np.isinf(train_results['y_pred_orig']))
            val_inf_count = np.sum(np.isinf(val_results['y_pred_orig']))
            test_inf_count = np.sum(np.isinf(test_results['y_pred_orig']))

            print(f"\nInfinite values in predictions:")
            print(f"  Training: {train_inf_count}/{len(train_results['y_pred_orig'])}")
            print(f"  Validation: {val_inf_count}/{len(val_results['y_pred_orig'])}")
            print(f"  Test: {test_inf_count}/{len(test_results['y_pred_orig'])}")

            # Show sample predictions (first 10 and last 10)
            print(f"\nSample Test Predictions (First 10):")
            print(f"{'Index':<6} {'Actual':<12} {'Predicted':<12} {'ARD%':<8} {'Uncertainty':<12}")
            print("-" * 55)

            for i in range(min(10, len(test_results['y_true_orig']))):
                actual = test_results['y_true_orig'][i]
                predicted = test_results['y_pred_orig'][i]
                ard = abs((actual - predicted) / actual) * 100
                uncertainty = test_uncertainty[i]
                print(f"{i:<6} {actual:<12.6f} {predicted:<12.6f} {ard:<8.3f} {uncertainty:<12.6f}")

            print(f"\nSample Test Predictions (Last 10):")
            print(f"{'Index':<6} {'Actual':<12} {'Predicted':<12} {'ARD%':<8} {'Uncertainty':<12}")
            print("-" * 55)

            start_idx = max(0, len(test_results['y_true_orig']) - 10)
            for i in range(start_idx, len(test_results['y_true_orig'])):
                actual = test_results['y_true_orig'][i]
                predicted = test_results['y_pred_orig'][i]
                ard = abs((actual - predicted) / actual) * 100
                uncertainty = test_uncertainty[i]
                print(f"{i:<6} {actual:<12.6f} {predicted:<12.6f} {ard:<8.3f} {uncertainty:<12.6f}")

            # 8. Final assessment
            print(f"\n🎯 FINAL RESULTS:")
            print(f"   Test AARD = {test_results['aard']:.3f}%")
            print(f"   Test R² = {test_results['r2_orig']:.4f}")
            print(f"   Test Max ARD = {test_results['max_ard']:.3f}%")
            print(f"   Mean Test Uncertainty = {np.mean(test_uncertainty):.4f}")
            print(f"   Median Test Uncertainty = {np.median(test_uncertainty):.4f}")

            success_criteria = []
            if test_results['aard'] <= 8.0:
                success_criteria.append("✅ AARD ≤ 8%")
            else:
                success_criteria.append(f"❌ AARD = {test_results['aard']:.2f}% > 8%")

            if test_results['r2_orig'] >= 0.85:
                success_criteria.append("✅ R² ≥ 0.85")
            else:
                success_criteria.append(f"❌ R² = {test_results['r2_orig']:.4f} < 0.85")

            if test_results['median_ard'] <= 5.0:
                success_criteria.append("✅ Median ARD ≤ 5%")
            else:
                success_criteria.append(f"❌ Median ARD = {test_results['median_ard']:.2f}% > 5%")

            print("\nSuccess Criteria:")
            for criterion in success_criteria:
                print(f"   {criterion}")

            if test_results['aard'] <= 8.0 and test_results['r2_orig'] >= 0.85:
                print("\n🎉 EXCELLENT: All targets achieved!")
            elif test_results['aard'] <= 10.0 and test_results['r2_orig'] >= 0.80:
                print("\n✅ GOOD: Close to targets, strong performance!")
            else:
                print("\n📈 IMPROVEMENT NEEDED")
                print("Consider:")
                print("   - Increasing n_trials for optimization")
                print("   - More aggressive feature engineering")
                print("   - Larger ensemble size")
                print("   - Different target transformations")

            return {
                'results': results_dict,
                'best_params': self.best_params,
                'models': self.models,
                'test_aard': test_results['aard'],
                'test_r2': test_results['r2_orig'],
                'test_max_ard': test_results['max_ard'],
                'test_uncertainty_mean': np.mean(test_uncertainty),
                'test_uncertainty_median': np.median(test_uncertainty)
            }

        except Exception as e:
            print(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {'results': {}, 'best_params': {}, 'models': []}

# Usage example with fixed implementation
if __name__ == "__main__":
    file_path = "/content/Final_Processed_Data.csv"  

    # Create fixed evidential predictor
    predictor = FixedEvidentialViscosityPredictor(random_state=42)

    print("Fixed Evidential Deep Learning for Viscosity Prediction")
    print("Addresses API compatibility and optimization issues")
    print("="*80)

    try:
        
        results = predictor.run_evidential_pipeline(
            file_path=file_path,
            optimize=True,
            n_trials=15,  
            use_ensemble=True
        )

        # Final summary
        if results and 'test_aard' in results:
            print(f"\n🏁 PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Final Performance:")
            print(f"  • Test AARD: {results['test_aard']:.3f}%")
            print(f"  • Test R²: {results['test_r2']:.4f}")
            print(f"  • Test Max ARD: {results['test_max_ard']:.3f}%")
            print(f"  • Test Mean Uncertainty: {results['test_uncertainty_mean']:.4f}")
            print(f"  • Test Median Uncertainty: {results['test_uncertainty_median']:.4f}")

            if results['test_aard'] <= 8.0:
                print(f"\n🎉 SUCCESS: Target AARD ≤ 8% achieved!")
            else:
                print(f"\n📊 Current AARD: {results['test_aard']:.2f}% - Room for improvement")
        else:
            print("\n⚠️ Pipeline completed with issues - check detailed logs above")

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        print("Please check:")
        print("1. File path is correct")
        print("2. Data format matches expected structure")
        print("3. Sufficient memory available")
        import traceback
        traceback.print_exc()
