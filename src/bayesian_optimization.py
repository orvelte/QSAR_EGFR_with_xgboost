#!/usr/bin/env python3
"""
Bayesian Optimization for XGBoost Hyperparameter Tuning
Uses scikit-optimize (skopt) to find optimal hyperparameters for QSAR modeling
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from skopt import gp_minimize
from skopt.space import Real, Integer
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
DATA_DIR = Path("data_proc")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

def load_data():
    """Load the processed data"""
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")
    meta = pd.read_csv(DATA_DIR / "meta.csv")
    
    # Create splits
    train_mask = meta["split"] == "train"
    val_mask = meta["split"] == "val"
    test_mask = meta["split"] == "test"
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Data shapes:")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val:   {X_val.shape}, {y_val.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def evaluate_model(X_train, y_train, X_val, y_val, params):
    """Evaluate model with given parameters"""
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0  # Suppress XGBoost output
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate validation RMSE (what we want to minimize)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    return val_rmse

# Define the search space for hyperparameters
dimensions = [
    Integer(100, 2000, name='n_estimators'),           # Number of boosting rounds
    Real(0.005, 0.1, name='learning_rate'),           # Learning rate
    Integer(3, 8, name='max_depth'),                  # Maximum tree depth
    Real(1.0, 10.0, name='min_child_weight'),         # Minimum child weight
    Real(0.5, 0.95, name='subsample'),                # Subsample ratio
    Real(0.5, 0.95, name='colsample_bytree'),         # Feature sampling ratio
    Real(0.0, 1.0, name='reg_alpha'),                 # L1 regularization
    Real(0.1, 20.0, name='reg_lambda'),               # L2 regularization
]

def run_bayesian_optimization(X_train, y_train, X_val, y_val, n_calls=50):
    """Run Bayesian optimization to find best hyperparameters"""
    print("Starting Bayesian Optimization...")
    print(f"Search space dimensions: {len(dimensions)}")
    print(f"Number of optimization calls: {n_calls}")
    print("=" * 60)
    
    # Create objective function with data
    def objective_with_data(params):
        # Convert list to dict
        param_dict = {
            'n_estimators': int(params[0]),
            'learning_rate': params[1],
            'max_depth': int(params[2]),
            'min_child_weight': params[3],
            'subsample': params[4],
            'colsample_bytree': params[5],
            'reg_alpha': params[6],
            'reg_lambda': params[7]
        }
        try:
            val_rmse = evaluate_model(X_train, y_train, X_val, y_val, param_dict)
            print(f"Params: {param_dict}")
            print(f"Val RMSE: {val_rmse:.4f}")
            print("-" * 50)
            return val_rmse
        except Exception as e:
            print(f"Error with params {param_dict}: {e}")
            return 1000.0  # Return high value for failed evaluations
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=objective_with_data,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=RANDOM_STATE,
        acq_func='EI',  # Expected Improvement acquisition function
        n_initial_points=10,  # Number of random initial points
        verbose=True
    )
    
    return result

def train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    """Train final model with best parameters and evaluate on all splits"""
    print("\nTraining final model with best parameters...")
    
    model = XGBRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions on all splits
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    def calc_metrics(y_true, y_pred):
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    metrics = {
        'train': calc_metrics(y_train, y_train_pred),
        'val': calc_metrics(y_val, y_val_pred),
        'test': calc_metrics(y_test, y_test_pred)
    }
    
    # Save model and results
    model.save_model(MODEL_DIR / "xgb_optimized.json")
    
    # Save metrics
    with open(REPORT_DIR / "metrics_xgb_optimized.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save best parameters
    with open(REPORT_DIR / "best_params_xgb.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Save predictions
    predictions = pd.DataFrame({
        'split': ['train'] * len(y_train) + ['val'] * len(y_val) + ['test'] * len(y_test),
        'y_true': np.concatenate([y_train, y_val, y_test]),
        'y_pred': np.concatenate([y_train_pred, y_val_pred, y_test_pred])
    })
    predictions.to_csv(REPORT_DIR / "predictions_xgb_optimized.csv", index=False)
    
    return model, metrics

def main():
    """Main function"""
    print("Bayesian Optimization for XGBoost Hyperparameter Tuning")
    print("=" * 60)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # Run Bayesian optimization
    result = run_bayesian_optimization(X_train, y_train, X_val, y_val, n_calls=50)
    
    # Extract best parameters
    best_params = {
        'n_estimators': int(result.x[0]),
        'learning_rate': result.x[1],
        'max_depth': int(result.x[2]),
        'min_child_weight': result.x[3],
        'subsample': result.x[4],
        'colsample_bytree': result.x[5],
        'reg_alpha': result.x[6],
        'reg_lambda': result.x[7]
    }
    
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best validation RMSE: {result.fun:.4f}")
    
    # Train final model with best parameters
    model, metrics = train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params)
    
    # Print final results
    print(f"\nFinal Model Performance:")
    for split, metric_dict in metrics.items():
        print(f"  {split}: RMSE={metric_dict['rmse']:.3f}, MAE={metric_dict['mae']:.3f}, R²={metric_dict['r2']:.3f}")
    
    print(f"\nResults saved to:")
    print(f"  Model: {MODEL_DIR / 'xgb_optimized.json'}")
    print(f"  Metrics: {REPORT_DIR / 'metrics_xgb_optimized.json'}")
    print(f"  Best params: {REPORT_DIR / 'best_params_xgb.json'}")
    print(f"  Predictions: {REPORT_DIR / 'predictions_xgb_optimized.csv'}")

if __name__ == "__main__":
    main()
