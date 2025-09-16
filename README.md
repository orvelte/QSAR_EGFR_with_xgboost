# QSAR Project: EGFR Inhibitor Activity Prediction

A comprehensive Quantitative Structure-Activity Relationship (QSAR) modeling project for predicting EGFR (Epidermal Growth Factor Receptor) inhibitor activity using machine learning approaches.

## Project Overview

This project implements a complete QSAR modeling pipeline to predict the biological activity of EGFR inhibitors. The workflow includes data collection from ChEMBL, molecular standardization, feature engineering with Morgan fingerprints, scaffold-based data splitting, and machine learning model development with XGBoost and Bayesian optimization.

### Key Results
- Dataset: 1,000 EGFR inhibitors with IC50/Ki values
- Features: 1,163 Morgan fingerprints (filtered from 2,048)
- Best Model: XGBoost with Bayesian optimization (R² = 0.415 on test set)
- Split Method: Scaffold-based splitting for realistic generalization assessment

## Project Structure

```
qsar_project/
├── data_raw/                 # Raw data from ChEMBL
│   └── egfr_raw.csv         # Initial EGFR activity data
├── data_proc/               # Processed and featurized data
│   ├── egfr_clean.csv       # Cleaned and standardized compounds
│   ├── X.npy               # Morgan fingerprint features
│   ├── y.npy               # Target values (pActivity)
│   └── meta.csv            # Metadata with splits and scaffolds
├── models/                  # Trained models
│   ├── xgb_baseline.json   # Baseline XGBoost model
│   └── xgb_optimized.json  # Bayesian-optimized XGBoost model
├── reports/                 # Results and analysis
│   ├── RESULTS.md          # Comprehensive results summary
│   ├── metrics_.json      # Model performance metrics
│   ├── predictions_.csv   # Model predictions
│   └── .png              # Visualization plots
├── src/                    # Source code
│   ├── fetch_chembl.py     # Data collection from ChEMBL
│   ├── clean_standardize.py # Molecular standardization
│   ├── featurize_split.py  # Feature engineering and splitting
│   ├── train_baseline.py   # Baseline model training
│   ├── bayesian_optimization.py # Hyperparameter optimization
│   └── utils_scaffold.py   # Scaffold utilities
├── environment.yml         # Conda environment specification
└── README.md              # This file
```

## Methods and Pipeline

### 1. Data Collection (`fetch_chembl.py`)
- Source: ChEMBL database (target: CHEMBL203 - EGFR)
- Criteria: IC50 and Ki values in nM units
- Output: 14,835 raw compound-activity pairs
- Filtering: Only compounds with valid SMILES and numeric activity values

### 2. Molecular Standardization (`clean_standardize.py`)
- Salt Stripping: Remove counterions using RDKit's SaltRemover
- Canonicalization: Generate canonical SMILES representations
- Deduplication: Aggregate duplicate compounds by median activity
- Activity Conversion: Transform nM values to pActivity = 9 - log₁₀(nM)
- Stratified Sampling: Balance activity distribution across quantiles
- Output: 1,000 standardized compounds with balanced activity range

### 3. Feature Engineering (`featurize_split.py`)
- Morgan Fingerprints: Generate 2,048-bit circular fingerprints (radius=2)
- Rare Bit Filtering: Remove features with <5 positive examples (43.2% reduction)
- Final Features: 1,163 informative molecular descriptors
- Scaffold Generation: Extract Murcko scaffolds for splitting

### 4. Data Splitting Strategy
- Method: Scaffold-based splitting (80/10/10 train/val/test)
- Rationale: Ensures test set contains novel molecular scaffolds
- Implementation: Group compounds by scaffold, then split scaffold groups
- Validation: Mean Tanimoto similarity between test and train = 0.523

### 5. Model Development

#### Baseline XGBoost (`train_baseline.py`)
- Algorithm: XGBoost Regressor with default parameters
- Regularization: Increased to reduce overfitting
- Evaluation: RMSE, MAE, R² on train/val/test splits
- Interpretability: SHAP analysis for feature importance

#### Bayesian Optimization (`bayesian_optimization.py`)
- Method: Gaussian Process-based optimization (scikit-optimize)
- Parameters: 8 hyperparameters (n_estimators, learning_rate, max_depth, etc.)
- Search Space: Carefully defined ranges for each parameter
- Evaluation: 50 optimization calls with 10 random initial points
- Result: Optimal hyperparameters for improved generalization

### 6. Model Evaluation
- Metrics: RMSE, MAE, R² for comprehensive assessment
- Visualizations: Parity plots, residual analysis, SHAP plots
- Error Analysis: Identification of worst prediction errors
- Generalization: Train-validation gap analysis

## Installation and Setup

### Prerequisites
- Python 3.9+
- Conda package manager

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate qsar_env

# Verify installation
python -c "import rdkit, xgboost, skopt; print('All packages installed successfully')"
```

### Dependencies
Key packages included in `environment.yml`:
- RDKit: Molecular informatics and cheminformatics
- XGBoost: Gradient boosting machine learning
- scikit-optimize: Bayesian optimization
- pandas/numpy: Data manipulation
- matplotlib/seaborn: Visualization
- SHAP: Model interpretability

## Usage

### Complete Pipeline Execution
```bash
# 1. Fetch data from ChEMBL
python src/fetch_chembl.py

# 2. Clean and standardize molecules
python src/clean_standardize.py

# 3. Generate features and split data
python src/featurize_split.py

# 4. Train baseline model
python src/train_baseline.py

# 5. Optimize hyperparameters
python src/bayesian_optimization.py
```

### Individual Script Usage
Each script can be run independently if the required input files are present:

```bash
# Train baseline model only
python src/train_baseline.py

# Run Bayesian optimization only
python src/bayesian_optimization.py
```

## Results Summary

### Model Performance
| Model | Train R² | Val R² | Test R² | Key Insight |
|-------|----------|--------|---------|-------------|
| XGBoost Baseline | 0.890 | 0.215 | 0.441 | Severe overfitting |
| XGBoost Optimized | 0.753 | 0.288 | 0.415 | Best overall |

### Key Findings
- Feature Efficiency: 43.2% feature reduction maintained performance
- Scaffold Diversity: Novel scaffolds remain challenging to predict
- Overfitting Control: Bayesian optimization improved generalization
- Model Robustness: XGBoost handles scaffold diversity well

## Technical Details

### Molecular Representations
- SMILES: Canonical molecular structure representations
- Morgan Fingerprints: 2,048-bit circular fingerprints (radius=2)
- Scaffolds: Murcko scaffolds for structural diversity assessment

### Machine Learning Approach
- Algorithm: XGBoost (gradient boosting)
- Optimization: Bayesian optimization with Gaussian processes
- Regularization: L1/L2 regularization, subsampling, feature sampling
- Validation: Scaffold-based splitting for realistic assessment

### Data Quality
- Source: ChEMBL (curated chemical database)
- Standardization: RDKit-based molecular cleaning
- Filtering: Activity range 3-12 pActivity units
- Balance: Stratified sampling across activity quantiles

## File Descriptions

### Data Files
- `egfr_raw.csv`: Raw ChEMBL data (14,835 compounds)
- `egfr_clean.csv`: Standardized data (1,000 compounds)
- `X.npy`: Feature matrix (1,000 × 1,163)
- `y.npy`: Target values (1,000 pActivity values)
- `meta.csv`: Metadata with splits and scaffolds

### Model Files
- `xgb_baseline.json`: Baseline XGBoost model
- `xgb_optimized.json`: Bayesian-optimized XGBoost model

### Results Files
- `RESULTS.md`: Comprehensive results summary
- `metrics_.json`: Performance metrics
- `predictions_.csv`: Model predictions
- `.png`: Visualization plots (parity, residuals, SHAP)

## Contributing

This project demonstrates a complete QSAR modeling pipeline. Key areas for extension:
- Additional molecular descriptors (3D, pharmacophore)
- Deep learning approaches (with larger datasets)
- Multi-target modeling
- Active learning for data collection

## License

This project is for educational and research purposes. Please cite appropriately if used in research.

## Contact

For questions about this QSAR modeling pipeline, please refer to the code documentation and results in the `reports/` directory.# QSAR_EGFR_with_xgboost
