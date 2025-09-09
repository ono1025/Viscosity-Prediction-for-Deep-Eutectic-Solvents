# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import requests
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import zscore, boxcox
import shap

# Chemistry imports (install with: pip install rdkit-pypi mordred)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from mordred import Calculator, descriptors
    print("✅ Chemistry libraries loaded successfully!")
except ImportError as e:
    print(f"❌ Error importing chemistry libraries: {e}")
    print("Please install: pip install rdkit-pypi mordred")

print("="*80)
print("COMPLETE CHEMISTRY DATASET PIPELINE")
print("="*80)

# =============================================================================
# PART 1: CHEMICAL DESCRIPTOR GENERATION
# =============================================================================

class ChemicalDescriptorGenerator:
    """Generate chemical descriptors using Mordred and RDKit"""
    
    def __init__(self):
        """Initialize the descriptor calculator"""
        try:
            self.calc = Calculator(descriptors, ignore_3D=False)
            print("🧪 Chemical descriptor calculator initialized")
        except:
            print("❌ Failed to initialize descriptor calculator")
            self.calc = None
    
    def cas_to_smiles(self, cas: str, timeout: int = 10) -> str:
        """Convert CAS number to SMILES string using online APIs"""
        if not cas or pd.isna(cas):
            return None
            
        try:
            # Try PubChem first
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES/JSON"
            response = requests.get(pubchem_url, timeout=timeout)
            
            if response.ok:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    return props[0]["CanonicalSMILES"]
            
            # Fallback to CACTUS
            cactus_url = f"https://cactus.nci.nih.gov/chemical/structure/{cas}/smiles"
            response = requests.get(cactus_url, timeout=timeout)
            
            if response.ok and "Not Found" not in response.text:
                return response.text.strip()
                
        except Exception as e:
            print(f"Warning: Failed to convert CAS {cas} to SMILES: {e}")
        
        return None
    
    def generate_descriptors(self, smiles: str) -> dict:
        """Generate Mordred descriptors from SMILES string"""
        if not smiles or not self.calc:
            return None
            
        try:
            # Create molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            
            # Calculate descriptors
            result = self.calc(mol)
            return result.asdict()
            
        except Exception as e:
            print(f"Warning: Failed to generate descriptors for SMILES {smiles}: {e}")
            return None
    
    def process_cas_list(self, cas_list: list, desc_type: str = "compound") -> dict:
        """Process a list of CAS numbers and generate descriptors"""
        print(f"\n🔬 Generating {desc_type} descriptors...")
        print(f"   Processing {len(cas_list)} unique CAS numbers")
        
        descriptors_dict = {}
        failed_count = 0
        
        for cas in tqdm(cas_list, desc=f"Processing {desc_type}"):
            # Convert CAS to SMILES
            smiles = self.cas_to_smiles(cas)
            
            if smiles:
                # Generate descriptors
                descriptors = self.generate_descriptors(smiles)
                if descriptors:
                    descriptors_dict[cas] = descriptors
                else:
                    failed_count += 1
            else:
                failed_count += 1
        
        success_rate = (len(descriptors_dict) / len(cas_list)) * 100
        print(f"   ✅ Success: {len(descriptors_dict)}/{len(cas_list)} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {failed_count}")
        
        return descriptors_dict

def generate_chemical_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to generate chemical descriptors for HBA and HBD compounds
    
    Args:
        df: DataFrame with HBA_CAS and HBD_CAS columns
        
    Returns:
        DataFrame with chemical descriptors added
    """
    print("\n" + "="*60)
    print("PART 1: CHEMICAL DESCRIPTOR GENERATION")
    print("="*60)
    
    # Initialize descriptor generator
    desc_gen = ChemicalDescriptorGenerator()
    if desc_gen.calc is None:
        print("❌ Cannot proceed without descriptor calculator")
        return df
    
    # Get unique CAS numbers
    hba_cas_list = df["HBA_CAS"].dropna().unique().tolist()
    hbd_cas_list = df["HBD_CAS"].dropna().unique().tolist()
    all_cas = list(set(hba_cas_list + hbd_cas_list))
    
    print(f"📊 Dataset Analysis:")
    print(f"   • Total rows: {len(df):,}")
    print(f"   • Unique HBA CAS: {len(hba_cas_list):,}")
    print(f"   • Unique HBD CAS: {len(hbd_cas_list):,}")
    print(f"   • Total unique CAS: {len(all_cas):,}")
    
    # Generate descriptors for all CAS numbers
    descriptors_by_cas = desc_gen.process_cas_list(all_cas, "chemical")
    
    if not descriptors_by_cas:
        print("❌ No descriptors generated!")
        return df
    
    # Convert to DataFrame
    df_descriptors = pd.DataFrame.from_dict(descriptors_by_cas, orient='index')
    df_descriptors.index.name = "CAS"
    
    print(f"\n📈 Descriptor Generation Results:")
    print(f"   • Descriptors generated: {df_descriptors.shape[1]:,}")
    print(f"   • Compounds processed: {df_descriptors.shape[0]:,}")
    print(f"   • Memory usage: {df_descriptors.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Create separate feature sets for HBA and HBD with prefixes
    hba_features = df_descriptors.add_prefix("HBA_").reset_index().rename(columns={"CAS": "HBA_CAS"})
    hbd_features = df_descriptors.add_prefix("HBD_").reset_index().rename(columns={"CAS": "HBD_CAS"})
    
    print(f"\n🔗 Merging descriptors with original dataset...")
    
    # Merge with original dataset
    df_merged = df.copy()
    df_merged = df_merged.merge(hba_features, on="HBA_CAS", how="left")
    df_merged = df_merged.merge(hbd_features, on="HBD_CAS", how="left")
    
    print(f"   • Original shape: {df.shape}")
    print(f"   • Merged shape: {df_merged.shape}")
    print(f"   • New columns added: {df_merged.shape[1] - df.shape[1]:,}")
    
    # Save intermediate result
    try:
        df_merged.to_csv("merged_dataset_with_mordred.csv", index=False)
        df_merged.to_excel("merged_dataset_with_mordred.xlsx", index=False)
        print(f"   ✅ Intermediate files saved:")
        print(f"      • merged_dataset_with_mordred.csv")
        print(f"      • merged_dataset_with_mordred.xlsx")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not save intermediate files: {e}")
    
    return df_merged

# =============================================================================
# PART 2: DATA PREPARATION AND CLEANING
# =============================================================================

class DataPreparationPipeline:
    """Complete data preparation and cleaning pipeline"""
    
    def __init__(self, target_col: str = "Viscosity"):
        self.target_col = target_col
        self.removed_columns = []
        self.transformation_applied = None
    
    def remove_high_missing_columns(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """Remove columns with more than threshold% missing values"""
        print(f"\n🧹 STEP 1: REMOVING HIGH MISSING VALUE COLUMNS (>{threshold*100}%)")
        print("-" * 60)
        
        initial_cols = df.shape[1]
        
        # Calculate missing percentages
        missing_percent = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percent[missing_percent > threshold * 100]
        
        # Don't remove target column
        if self.target_col in high_missing_cols.index:
            high_missing_cols = high_missing_cols.drop(self.target_col)
            print(f"⚠️  Target column '{self.target_col}' has high missing values but will be kept!")
        
        if len(high_missing_cols) > 0:
            print(f"📋 Found {len(high_missing_cols)} columns with >{threshold*100}% missing values:")
            
            # Show worst offenders
            worst_missing = high_missing_cols.sort_values(ascending=False).head(10)
            for col, pct in worst_missing.items():
                print(f"   • {col}: {pct:.1f}% missing")
            
            if len(high_missing_cols) > 10:
                print(f"   • ... and {len(high_missing_cols)-10} more columns")
            
            # Remove columns
            df_cleaned = df.drop(columns=high_missing_cols.index)
            self.removed_columns.extend(high_missing_cols.index.tolist())
            
            print(f"\n✅ RESULT:")
            print(f"   • Columns removed: {len(high_missing_cols)}")
            print(f"   • Columns remaining: {df_cleaned.shape[1]} (was {initial_cols})")
            
            return df_cleaned
        else:
            print("✅ No columns found with >10% missing values!")
            return df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names while preserving target"""
        print("\n🏷️  STEP 2: CLEANING COLUMN NAMES")
        print("-" * 40)
        
        original_names = df.columns.tolist()
        target_index = original_names.index(self.target_col) if self.target_col in original_names else -1
        
        # Clean names
        cleaned_names = []
        name_changes = []
        
        for i, original_name in enumerate(original_names):
            if i == target_index:
                
                clean_name = 'Viscosity'
            else:
                # Clean other names
                clean_name = str(original_name).lower().strip()
                clean_name = ''.join(c if c.isalnum() else '_' for c in clean_name)
                clean_name = '_'.join(word for word in clean_name.split('_') if word)
                
                # Handle duplicates
                base_name = clean_name
                counter = 1
                while clean_name in cleaned_names:
                    clean_name = f"{base_name}_{counter}"
                    counter += 1
            
            cleaned_names.append(clean_name)
            
            if original_name != clean_name:
                name_changes.append((original_name, clean_name))
        
        # Apply changes
        df.columns = cleaned_names
        self.target_col = 'Viscosity'  
        
        print(f"✅ Column names cleaned: {len(name_changes)}")
        print(f"   • Target column preserved as: '{self.target_col}'")
        
        return df
    
    def remove_single_value_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with only one unique value"""
        print("\n🗑️  STEP 3: REMOVING SINGLE-VALUE COLUMNS")
        print("-" * 50)
        
        initial_cols = df.shape[1]
        single_value_cols = []
        
        for col in df.columns:
            if col != self.target_col:  
                unique_count = df[col].nunique()
                if unique_count <= 1:
                    single_value_cols.append(col)
        
        if len(single_value_cols) > 0:
            print(f"📋 Found {len(single_value_cols)} columns with ≤1 unique value")
            
            df_cleaned = df.drop(columns=single_value_cols)
            self.removed_columns.extend(single_value_cols)
            
            print(f"✅ Columns removed: {len(single_value_cols)}")
            print(f"   • Columns remaining: {df_cleaned.shape[1]} (was {initial_cols})")
            
            return df_cleaned
        else:
            print("✅ No single-value columns found!")
            return df
    
    def remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate columns"""
        print("\n🔄 STEP 4: REMOVING DUPLICATE COLUMNS")
        print("-" * 45)
        
        initial_cols = df.shape[1]
        columns_to_remove = set()
        
        # Check for duplicates
        for i in range(len(df.columns)):
            for j in range(i+1, len(df.columns)):
                col1, col2 = df.columns[i], df.columns[j]
                if col1 != self.target_col and col2 != self.target_col:  
                    if df[col1].equals(df[col2]):
                        columns_to_remove.add(col2)  
        
        if len(columns_to_remove) > 0:
            print(f"📋 Found {len(columns_to_remove)} duplicate columns")
            
            df_cleaned = df.drop(columns=list(columns_to_remove))
            self.removed_columns.extend(list(columns_to_remove))
            
            print(f"✅ Duplicate columns removed: {len(columns_to_remove)}")
            print(f"   • Columns remaining: {df_cleaned.shape[1]} (was {initial_cols})")
            
            return df_cleaned
        else:
            print("✅ No duplicate columns found!")
            return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme outliers in target variable"""
        print(f"\n📊 STEP 5: HANDLING OUTLIERS")
        print("-" * 35)
        
        if self.target_col not in df.columns:
            print(f"❌ Target column '{self.target_col}' not found!")
            return df
        
        initial_rows = len(df)
        target_data = df[self.target_col].dropna()
        
        # Calculate IQR
        Q1 = target_data.quantile(0.25)
        Q3 = target_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        upper_bound = Q3 + 1.5 * IQR
        
        # Get extreme outliers
        high_outliers = df[df[self.target_col] > upper_bound]
        
        print(f"📈 Outlier Analysis:")
        print(f"   • Q1: {Q1:.2f}")
        print(f"   • Q3: {Q3:.2f}")
        print(f"   • IQR: {IQR:.2f}")
        print(f"   • Upper bound: {upper_bound:.2f}")
        print(f"   • High outliers found: {len(high_outliers)}")
        
        # Remove only top 20 extreme outliers
        if len(high_outliers) > 0:
            outliers_sorted = high_outliers.sort_values(by=self.target_col, ascending=False)
            to_remove = min(20, len(outliers_sorted))
            to_drop = outliers_sorted.head(to_remove).index
            
            df_cleaned = df.drop(index=to_drop).reset_index(drop=True)
            
            print(f"✅ Removed top {to_remove} extreme outliers")
            print(f"   • Rows remaining: {len(df_cleaned)} (was {initial_rows})")
            
            return df_cleaned
        else:
            print("✅ No extreme outliers to remove")
            return df
    
    def handle_categorical_and_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical encoding and missing values"""
        print("\n🏷️  STEP 6: HANDLING CATEGORICAL & MISSING VALUES")
        print("-" * 55)
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            print(f"📋 Found {len(categorical_cols)} categorical columns")
            df_encoded = df.copy()
            
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    # One-hot encode
                    encoded = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                    df_encoded = df_encoded.drop(columns=[col])
                    df_encoded = pd.concat([df_encoded, encoded], axis=1)
                    print(f"   • {col}: One-hot encoded ({encoded.shape[1]} columns)")
                else:
                    # Label encode
                    le = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    df_encoded = df_encoded.drop(columns=[col])
                    print(f"   • {col}: Label encoded")
            
            df = df_encoded
        else:
            print("✅ No categorical columns found")
        
        # Handle missing values
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            print(f"\n📋 Handling missing values in {len(missing_cols)} columns")
            
            for col in missing_cols.index:
                if col == self.target_col:
                    continue  # Skip target column
                
                if df[col].dtype in ['object']:
                    mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                else:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
        
        print(f"✅ Missing values handled")
        print(f"   • Remaining missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def analyze_and_transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze target distribution and apply transformation if needed"""
        print(f"\n📈 STEP 7: TARGET VARIABLE ANALYSIS & TRANSFORMATION")
        print("-" * 60)
        
        if self.target_col not in df.columns:
            print(f"❌ Target column '{self.target_col}' not found!")
            return df
        
        target_data = df[self.target_col].dropna()
        
        print(f"🎯 Target Variable '{self.target_col}' Analysis:")
        print(f"   • Count: {len(target_data):,} samples")
        print(f"   • Range: {target_data.min():.2f} to {target_data.max():,.2f}")
        print(f"   • Variation Factor: {target_data.max()/target_data.min():.1f}x")
        print(f"   • Mean: {target_data.mean():.2f}")
        print(f"   • Median: {target_data.median():.2f}")
        
        # Distribution analysis
        skewness = target_data.skew()
        kurtosis = target_data.kurtosis()
        
        print(f"\n📊 Distribution Analysis:")
        print(f"   • Skewness: {skewness:.3f} {'(highly right-skewed!)' if skewness > 2 else '(right-skewed)' if skewness > 1 else '(roughly normal)'}")
        print(f"   • Kurtosis: {kurtosis:.3f}")
        
        # Check for transformation need
        needs_transform = target_data.max() / target_data.min() > 100 or abs(skewness) > 2
        
        if needs_transform:
            print(f"\n🔧 TRANSFORMATION NEEDED!")
            print(f"   • Reason: Wide range ({target_data.max()/target_data.min():.1f}x) and/or high skewness ({skewness:.2f})")
            
            # Test transformations
            transformations = {}
            
            # Log transformation (if all positive)
            if target_data.min() > 0:
                log_transformed = np.log(target_data)
                transformations['log'] = {
                    'data': log_transformed,
                    'skewness': log_transformed.skew(),
                    'range_factor': log_transformed.max() / log_transformed.min()
                }
                print(f"   • Log transform: skewness = {log_transformed.skew():.3f}")
            
            # Square root transformation
            sqrt_transformed = np.sqrt(target_data)
            transformations['sqrt'] = {
                'data': sqrt_transformed,
                'skewness': sqrt_transformed.skew(),
                'range_factor': sqrt_transformed.max() / sqrt_transformed.min()
            }
            print(f"   • Sqrt transform: skewness = {sqrt_transformed.skew():.3f}")
            
            # Box-Cox transformation
            try:
                boxcox_transformed, lambda_param = boxcox(target_data)
                transformations['boxcox'] = {
                    'data': boxcox_transformed,
                    'skewness': boxcox_transformed.skew(),
                    'lambda': lambda_param
                }
                print(f"   • Box-Cox transform (λ={lambda_param:.3f}): skewness = {boxcox_transformed.skew():.3f}")
            except:
                pass
            
            # Choose best transformation
            best_transform = min(transformations.keys(),
                               key=lambda x: abs(transformations[x]['skewness']))
            
            print(f"\n🏆 BEST TRANSFORMATION: {best_transform.upper()}")
            
            # Apply transformation
            df_transformed = df.copy()
            
            if best_transform == 'log':
                df_transformed[f'{self.target_col}_log'] = np.log(df[self.target_col])
                new_target_col = f'{self.target_col}_log'
            elif best_transform == 'sqrt':
                df_transformed[f'{self.target_col}_sqrt'] = np.sqrt(df[self.target_col])
                new_target_col = f'{self.target_col}_sqrt'
            elif best_transform == 'boxcox':
                lambda_param = transformations['boxcox']['lambda']
                df_transformed[f'{self.target_col}_boxcox'] = boxcox(df[self.target_col], lmbda=lambda_param)
                new_target_col = f'{self.target_col}_boxcox'
            
            self.target_col = new_target_col
            self.transformation_applied = best_transform
            
            print(f"✅ New target column: '{new_target_col}'")
            
            return df_transformed
        else:
            print(f"✅ NO TRANSFORMATION NEEDED")
            return df
    
    def perform_feature_selection(self, df: pd.DataFrame, n_features: int = 100) -> tuple:
        """Perform feature selection to get top N features"""
        print(f"\n🎯 STEP 8: FEATURE SELECTION (Top {n_features} features)")
        print("-" * 55)
        
        # Prepare data
        df_clean = df.dropna(subset=[self.target_col]).copy()
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col != self.target_col]
        X = df_clean[feature_cols]
        y = df_clean[self.target_col]
        
        print(f"📊 Feature selection setup:")
        print(f"   • Available features: {len(feature_cols)}")
        print(f"   • Samples: {len(X)}")
        print(f"   • Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Only numeric features
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())
        
        print(f"   • Numeric features: {X.shape[1]}")
        
        # Feature selection methods
        all_feature_scores = {}
        
        # Method 1: Random Forest
        print(f"\n🌲 Random Forest Feature Importance")
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
            rf.fit(X, y)
            rf_scores = pd.Series(rf.feature_importances_, index=X.columns)
            all_feature_scores['rf'] = rf_scores
            print(f"   ✅ Completed!")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Method 2: F-Score
        print(f"\n📈 F-Score Selection")
        try:
            f_scores, _ = f_regression(X, y)
            f_scores = pd.Series(f_scores, index=X.columns)
            all_feature_scores['f_score'] = f_scores
            print(f"   ✅ Completed!")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # Method 3: Mutual Information
        print(f"\n🔗 Mutual Information")
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=X.columns)
            all_feature_scores['mi'] = mi_scores
            print(f"   ✅ Completed!")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        if not all_feature_scores:
            print("❌ All feature selection methods failed!")
            return df_clean, None
        
        print(f"\n🎯 COMBINING METHODS")
        print(f"   • Methods succeeded: {list(all_feature_scores.keys())}")
        
        # Normalize and combine scores
        normalized_scores = {}
        for method, scores in all_feature_scores.items():
            min_val, max_val = scores.min(), scores.max()
            if max_val > min_val:
                normalized_scores[method] = (scores - min_val) / (max_val - min_val)
        
        # Calculate ensemble average
        all_features = set()
        for scores in normalized_scores.values():
            all_features.update(scores.index)
        
        ensemble_scores = {}
        for feature in all_features:
            scores_list = []
            for method_scores in normalized_scores.values():
                if feature in method_scores.index:
                    scores_list.append(method_scores[feature])
            if scores_list:
                ensemble_scores[feature] = np.mean(scores_list)
        
        ensemble_scores = pd.Series(ensemble_scores).sort_values(ascending=False)
        
        # Select top features
        top_features = ensemble_scores.head(n_features).index.tolist()
        
        print(f"🏆 Top {len(top_features)} features selected!")
        
        # Create final dataset
        final_columns = top_features + [self.target_col]
        df_final = df[final_columns].copy()
        df_final = df_final.dropna(subset=[self.target_col])
        
        print(f"\n✅ FINAL DATASET:")
        print(f"   • Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
        print(f"   • Features: {df_final.shape[1] - 1}")
        
        return df_final, top_features

def prepare_dataset(df: pd.DataFrame, target_col: str = "Viscosity", n_features: int = 100) -> pd.DataFrame:
    """
    Main function to prepare dataset with full pipeline
    
    Args:
        df: Input DataFrame with chemical descriptors
        target_col: Name of target column
        n_features: Number of features to select
        
    Returns:
        Final prepared DataFrame
    """
    print("\n" + "="*60)
    print("PART 2: DATA PREPARATION AND CLEANING")
    print("="*60)
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline(target_col)
    
    original_shape = df.shape
    print(f"📊 Starting with: {original_shape[0]:,} rows × {original_shape[1]:,} columns")
    
    # Run pipeline steps
    df = pipeline.remove_high_missing_columns(df)
    df = pipeline.clean_column_names(df)
    df = pipeline.remove_single_value_columns(df)
    df = pipeline.remove_duplicate_columns(df)
    df = pipeline.handle_outliers(df)
    df = pipeline.handle_categorical_and_missing(df)
    df = pipeline.analyze_and_transform_target(df)
    
    # Feature selection
    df_final, selected_features = pipeline.perform_feature_selection(df, n_features)
    
    if df_final is None:
        print("❌ Pipeline failed!")
        return None
    
    # Save results
    try:
        csv_filename = "final_chemistry_dataset.csv"
        xlsx_filename = "final_chemistry_dataset.xlsx"
        
        df_final.to_csv(csv_filename, index=False)
        
        with pd.ExcelWriter(xlsx_filename, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='Final_Dataset', index=False)
            
            # Feature list
            if selected_features:
                feature_df = pd.DataFrame({
                    'Feature_Name': selected_features,
                    'Rank': range(1, len(selected_features) + 1)
                })
                feature_df.to_excel(writer, sheet_name='Selected_Features', index=False)
            
            # Summary
            summary_data = {
                'Metric': ['Original_Rows', 'Original_Columns', 'Final_Rows', 'Final_Columns', 
                          'Features_Selected', 'Target_Column', 'Transformation_Applied'],
                'Value': [original_shape[0], original_shape[1], df_final.shape[0], df_final.shape[1],
                         len(selected_features) if selected_features else 0, pipeline.target_col,
                         pipeline.transformation_applied or 'None']
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"\n💾 FILES SAVED:")
        print(f"   • {csv_filename}")
        print(f"   • {xlsx_filename}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save files: {e}")
    
    return df_final

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run_complete_pipeline(input_file: str, target_col: str = "Viscosity", n_features: int = 100):
    """
    Run the complete chemistry pipeline from start to finish
    
    Args:
        input_file: Path to input CSV file with HBA_CAS and HBD_CAS columns
        target_col: Name of target column (default: "Viscosity")
        n_features: Number of features to select (default: 100)
        
    Returns:
        Final processed DataFrame ready for machine learning
    """
    print("🚀 STARTING COMPLETE CHEMISTRY PIPELINE")
    print("="*80)
    print(f"📁 Input file: {input_file}")
    print(f"🎯 Target column: {target_col}")
    print(f"🔢 Features to select: {n_features}")
    print("="*80)
    
    # Load initial dataset
    try:
        if input_file.endswith('.csv'):
            df_original = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            df_original = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported file format! Use CSV or Excel.")
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   • Shape: {df_original.shape}")
        print(f"   • Columns: {list(df_original.columns)}")
        
        # Check required columns
        required_cols = ['HBA_CAS', 'HBD_CAS', target_col]
        missing_cols = [col for col in required_cols if col not in df_original.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # PART 1: Generate chemical descriptors
    print(f"\n⏳ Estimated time: 5-15 minutes depending on dataset size...")
    df_with_descriptors = generate_chemical_descriptors(df_original)
    
    if df_with_descriptors is None:
        print("❌ Chemical descriptor generation failed!")
        return None
    
    # PART 2: Prepare and clean dataset
    final_dataset = prepare_dataset(df_with_descriptors, target_col, n_features)
    
    if final_dataset is None:
        print("❌ Data preparation failed!")
        return None
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 TRANSFORMATION SUMMARY:")
    print(f"   • Original dataset: {df_original.shape[0]:,} rows × {df_original.shape[1]:,} columns")
    print(f"   • With descriptors: {df_with_descriptors.shape[0]:,} rows × {df_with_descriptors.shape[1]:,} columns")
    print(f"   • Final dataset: {final_dataset.shape[0]:,} rows × {final_dataset.shape[1]:,} columns")
    print(f"   • Selected features: {final_dataset.shape[1] - 1}")
    print(f"   • Target samples: {final_dataset.shape[0]:,}")
    
    print(f"\n✨ Dataset ready for machine learning!")
    print(f"📈 Expected performance: High-quality features selected from chemical descriptors")
    print(f"🎯 Use this dataset for viscosity prediction models")
    
    return final_dataset

# =============================================================================
# EXAMPLE USAGE AND CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the complete chemistry pipeline with flexible options
    """
    
    # Configuration
    ORIGINAL_FILE = "\content\Raw_dataset.csv"           
    MERGED_FILE = "\content\Dataset_with_all_descriptormerged_dataset_with_mordred.csv" 
    TARGET_COLUMN = "Viscosity"                     
    N_FEATURES = 100                                
    
    print("🧪 CHEMISTRY ML PIPELINE CONFIGURATION")
    print("-" * 50)
    print(f"Original file: {ORIGINAL_FILE}")
    print(f"Merged file: {MERGED_FILE}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Features to select: {N_FEATURES}")
    print()
    
    # Check which files exist
    original_exists = os.path.exists(ORIGINAL_FILE)
    merged_exists = os.path.exists(MERGED_FILE)
    
    print("📂 FILE STATUS CHECK:")
    print(f"   • Original file exists: {'✅' if original_exists else '❌'} {ORIGINAL_FILE}")
    print(f"   • Merged file exists: {'✅' if merged_exists else '❌'} {MERGED_FILE}")
    print()
    
    # Determine execution mode
    if merged_exists:
        print("🎯 EXECUTION OPTIONS:")
        print("   1. Use existing merged file (faster)")
        print("   2. Regenerate descriptors from original")
        print("   3. Run only data preparation")
        print("   4. Run only descriptor generation")
        print()
        
        # For automatic execution, choose option 1 (use existing merged file)
        print("🚀 RUNNING PIPELINE WITH EXISTING MERGED FILE...")
        final_df = run_complete_pipeline(
            original_file=ORIGINAL_FILE,
            merged_file=MERGED_FILE,
            target_col=TARGET_COLUMN,
            n_features=N_FEATURES,
            skip_descriptors=True  # Use existing merged file
        )
        
    elif original_exists:
        print("🚀 RUNNING COMPLETE PIPELINE FROM ORIGINAL FILE...")
        print("   This will generate descriptors first, then prepare data")
        print()
        
        final_df = run_complete_pipeline(
            original_file=ORIGINAL_FILE,
            merged_file=None,
            target_col=TARGET_COLUMN,
            n_features=N_FEATURES,
            skip_descriptors=False  # Generate descriptors
        )
        
    else:
        print("❌ MISSING INPUT FILES!")
        print("Please ensure you have one of the following:")
        print(f"   • {ORIGINAL_FILE} (for complete pipeline)")
        print(f"   • {MERGED_FILE} (for data preparation only)")
        print()
        print("Required columns in original file:")
        print("   • HBA_CAS: CAS numbers for hydrogen bond acceptors")
        print("   • HBD_CAS: CAS numbers for hydrogen bond donors")
        print(f"   • {TARGET_COLUMN}: Target variable for prediction")
        final_df = None
    
    # Display results if successful
    if final_df is not None:
        print("\n🔬 FINAL DATASET PREVIEW:")
        print("-" * 30)
        print(final_df.head())
        print("\n📊 Dataset Info:")
        print(final_df.info())
        
        print("\n🎯 Target Variable Statistics:")
        target_cols = [col for col in final_df.columns if 'viscosity' in col.lower()]
        if target_cols:
            target_stats = final_df[target_cols[0]].describe()
            print(target_stats)
        
        # Validate output
        validation_passed = validate_pipeline_output(final_df, N_FEATURES)
        
        if validation_passed:
            print("\n🎉 PIPELINE EXECUTION SUCCESSFUL!")
            print("✅ Dataset is ready for machine learning")
        else:
            print("\n⚠️  PIPELINE COMPLETED WITH WARNINGS")
            print("🔍 Please review the validation results above")
            
    else:
        print("❌ PIPELINE EXECUTION FAILED!")
        print("Please check the error messages above and fix the issues.")

# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS FOR TWO-DATASET WORKFLOW
# =============================================================================

def check_descriptor_quality(merged_file: str = "merged_dataset_with_mordred.csv") -> dict:
    """Check the quality of generated chemical descriptors"""
    
    try:
        df = pd.read_csv(merged_file)
        
        # Count HBA and HBD descriptors
        hba_cols = [col for col in df.columns if col.startswith('HBA_')]
        hbd_cols = [col for col in df.columns if col.startswith('HBD_')]
        
        # Check missing values
        hba_missing = df[hba_cols].isnull().sum().sum()
        hbd_missing = df[hbd_cols].isnull().sum().sum()
        
        # Check if descriptors were generated
        total_hba_descriptors = len(hba_cols)
        total_hbd_descriptors = len(hbd_cols)
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'hba_descriptors': total_hba_descriptors,
            'hbd_descriptors': total_hbd_descriptors,
            'hba_missing_values': hba_missing,
            'hbd_missing_values': hbd_missing,
            'descriptor_success_rate': ((total_hba_descriptors + total_hbd_descriptors) / (2 * 1826)) * 100 if total_hba_descriptors > 0 else 0
        }
        
        print("🔍 DESCRIPTOR QUALITY REPORT:")
        print(f"   • Total samples: {quality_report['total_rows']:,}")
        print(f"   • HBA descriptors: {quality_report['hba_descriptors']:,}")
        print(f"   • HBD descriptors: {quality_report['hbd_descriptors']:,}")
        print(f"   • HBA missing values: {quality_report['hba_missing_values']:,}")
        print(f"   • HBD missing values: {quality_report['hbd_missing_values']:,}")
        print(f"   • Descriptor success rate: {quality_report['descriptor_success_rate']:.1f}%")
        
        if quality_report['descriptor_success_rate'] > 80:
            print("✅ Descriptor quality is good!")
        elif quality_report['descriptor_success_rate'] > 50:
            print("⚠️  Descriptor quality is moderate")
        else:
            print("❌ Poor descriptor quality - consider regenerating")
            
        return quality_report
        
    except Exception as e:
        print(f"❌ Error checking descriptor quality: {e}")
        return None

def compare_datasets(original_file: str, merged_file: str, target_col: str = "Viscosity"):
    """Compare original and merged datasets"""
    
    print("📊 DATASET COMPARISON")
    print("-" * 25)
    
    try:
        # Load both datasets
        df_original = pd.read_csv(original_file)
        df_merged = pd.read_csv(merged_file)
        
        print(f"Original dataset:")
        print(f"   • Shape: {df_original.shape}")
        print(f"   • Target samples: {df_original[target_col].dropna().shape[0]:,}")
        print(f"   • Missing in target: {df_original[target_col].isnull().sum()}")
        
        print(f"\nMerged dataset:")
        print(f"   • Shape: {df_merged.shape}")
        print(f"   • Columns added: {df_merged.shape[1] - df_original.shape[1]:,}")
        print(f"   • Memory increase: {(df_merged.memory_usage(deep=True).sum() - df_original.memory_usage(deep=True).sum()) / 1024**2:.1f} MB")
        
        # Check if any samples were lost
        samples_lost = df_original.shape[0] - df_merged.shape[0]
        if samples_lost > 0:
            print(f"⚠️  Samples lost during merging: {samples_lost}")
        else:
            print(f"✅ No samples lost during merging")
            
        # Check descriptor coverage
        original_hba_unique = df_original['HBA_CAS'].nunique()
        original_hbd_unique = df_original['HBD_CAS'].nunique()
        
        hba_with_descriptors = df_merged['HBA_CAS'].dropna().nunique()
        hbd_with_descriptors = df_merged['HBD_CAS'].dropna().nunique()
        
        print(f"\nDescriptor coverage:")
        print(f"   • HBA: {hba_with_descriptors}/{original_hba_unique} ({(hba_with_descriptors/original_hba_unique)*100:.1f}%)")
        print(f"   • HBD: {hbd_with_descriptors}/{original_hbd_unique} ({(hbd_with_descriptors/original_hbd_unique)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error comparing datasets: {e}")

# =============================================================================
# UTILITY FUNCTIONS FOR POST-PROCESSING
# =============================================================================

def load_processed_dataset(filename: str = "final_chemistry_dataset.csv") -> pd.DataFrame:
    """Load the processed dataset for machine learning"""
    try:
        df = pd.read_csv(filename)
        print(f"✅ Processed dataset loaded: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading processed dataset: {e}")
        return None

def split_features_target(df: pd.DataFrame, target_col: str = "Viscosity") -> tuple:
    """Split dataset into features and target for ML"""
    if target_col not in df.columns:
        # Try to find transformed target column
        possible_targets = [col for col in df.columns if 'viscosity' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            print(f"Using target column: {target_col}")
        else:
            print("❌ No target column found!")
            return None, None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"✅ Dataset split:")
    print(f"   • Features (X): {X.shape}")
    print(f"   • Target (y): {y.shape}")
    
    return X, y

def get_pipeline_summary(summary_file: str = "final_chemistry_dataset.xlsx") -> dict:
    """Get summary of pipeline execution"""
    try:
        summary_df = pd.read_excel(summary_file, sheet_name='Summary')
        features_df = pd.read_excel(summary_file, sheet_name='Selected_Features')
        
        summary = {
            'pipeline_stats': dict(zip(summary_df['Metric'], summary_df['Value'])),
            'selected_features': features_df['Feature_Name'].tolist()
        }
        
        return summary
        
    except Exception as e:
        print(f"❌ Error loading pipeline summary: {e}")
        return None

# =============================================================================
# PIPELINE VALIDATION AND TESTING
# =============================================================================

def validate_pipeline_output(df: pd.DataFrame, expected_features: int = 100) -> bool:
    """Validate the pipeline output meets expectations"""
    print("\n🔍 VALIDATING PIPELINE OUTPUT")
    print("-" * 35)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Dataset not empty
    if len(df) > 0:
        print("✅ Dataset is not empty")
        checks_passed += 1
    else:
        print("❌ Dataset is empty!")
    
    # Check 2: Correct number of columns
    expected_cols = expected_features + 1  
    if df.shape[1] == expected_cols:
        print(f"✅ Correct number of columns ({expected_cols})")
        checks_passed += 1
    else:
        print(f"❌ Expected {expected_cols} columns, got {df.shape[1]}")
    
    # Check 3: No missing values in critical columns
    if df.isnull().sum().sum() == 0:
        print("✅ No missing values")
        checks_passed += 1
    else:
        print(f"⚠️  {df.isnull().sum().sum()} missing values found")
    
    # Check 4: Target column exists and has valid range
    target_cols = [col for col in df.columns if 'viscosity' in col.lower()]
    if target_cols:
        target_col = target_cols[0]
        target_range = df[target_col].max() - df[target_col].min()
        if target_range > 0:
            print(f"✅ Target column '{target_col}' has valid range")
            checks_passed += 1
        else:
            print(f"❌ Target column has no variation")
    else:
        print("❌ No target column found")
    
    # Check 5: Features are numeric
    numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
    if numeric_cols >= expected_features:
        print(f"✅ Sufficient numeric features ({numeric_cols})")
        checks_passed += 1
    else:
        print(f"❌ Only {numeric_cols} numeric features found")
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"\n📊 VALIDATION RESULT: {checks_passed}/{total_checks} checks passed ({success_rate:.0f}%)")
    
    if success_rate >= 80:
        print("🎉 Pipeline output is valid and ready for ML!")
        return True
    else:
        print("⚠️  Pipeline output may have issues")
        return False

print("\n" + "="*80)
print("📚 PIPELINE DOCUMENTATION")
print("="*80)
print("="*80)