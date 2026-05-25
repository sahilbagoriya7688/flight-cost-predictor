"""
Model Training Module for Flight Cost Predictor
Trains Linear Regression (baseline) and XGBoost regressor with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def split_data(df: pd.DataFrame, target_col: str = 'Price', test_size: float = 0.2):
      X = df.drop(columns=[target_col])
      y = df[target_col]
      return train_test_split(X, y, test_size=test_size, random_state=42)


def train_linear_regression(X_train, y_train):
      """Train a Linear Regression baseline model."""
      lr_model = LinearRegression()
      lr_model.fit(X_train, y_train)
      return lr_model


def train_xgboost(X_train, y_train, hyperparameter_tuning: bool = True):
      """Train XGBoost with optional GridSearchCV hyperparameter tuning."""
      if hyperparameter_tuning:
                param_grid = {
                              'n_estimators': [100, 200, 300],
                              'max_depth': [3, 5, 7],
                              'learning_rate': [0.05, 0.1, 0.2],
                              'subsample': [0.8, 1.0],
                              'colsample_bytree': [0.8, 1.0],
                }
                xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(
                    xgb_base, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                print(f"Best XGBoost params: {grid_search.best_params_}")
                return grid_search.best_estimator_
else:
        model = XGBRegressor(
                      n_estimators=300, max_depth=5, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
          model.fit(X_train, y_train)
        return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
      """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
              'model': model_name,
              'R2 Score': round(r2, 4),
              'MAE': round(mae, 2),
              'RMSE': round(rmse, 2),
    }
    print(f"\n{model_name} | R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return metrics


def save_model(model, filepath: str):
      """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def load_model(filepath: str):
      """Load model from disk."""
    return joblib.load(filepath)


def run_training_pipeline(df: pd.DataFrame, target_col: str = 'Price',
                                                    save_path: str = 'models/'):
                                                          """
                                                              Full training pipeline: split data, train LR and XGBoost,
                                                                  evaluate both, save models, return best model.
                                                                      """
                                                          os.makedirs(save_path, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(df, target_col)

    print("Training Linear Regression baseline...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    save_model(lr_model, f"{save_path}linear_regression.pkl")

    print("\nTraining XGBoost with hyperparameter tuning...")
    xgb_model = train_xgboost(X_train, y_train, hyperparameter_tuning=True)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Tuned)")
    save_model(xgb_model, f"{save_path}xgboost_model.pkl")

    joblib.dump(list(X_train.columns), f"{save_path}feature_names.pkl")

    best_model = xgb_model if xgb_metrics['R2 Score'] >= lr_metrics['R2 Score'] else lr_model
    best_name = 'XGBoost' if best_model is xgb_model else 'Linear Regression'
    print(f"\nBest model: {best_name} (R2={max(xgb_metrics['R2 Score'], lr_metrics['R2 Score'])})")

    return best_model, {'linear_regression': lr_metrics, 'xgboost': xgb_metrics}, X_test, y_test
