"""
EDA and Visualization Module for Flight Cost Predictor
Uses Seaborn and Matplotlib for exploratory data analysis plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid', palette='muted')


def plot_price_distribution(df, target_col='Price'):
    """Plot histogram and boxplot of flight prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[target_col], bins=50, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Flight Price Distribution')
    axes[0].set_xlabel('Price (INR)')
    sns.boxplot(y=df[target_col], ax=axes[1], color='lightcoral')
    axes[1].set_title('Price Boxplot')
    plt.tight_layout()
    return fig


def plot_airline_vs_price(df, airline_col='Airline', target_col='Price'):
    """Boxplot of price by airline."""
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby(airline_col)[target_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=airline_col, y=target_col, order=order, palette='Set2', ax=ax)
    ax.set_title('Flight Price by Airline')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_stops_vs_price(df, stops_col='Total_Stops', target_col='Price'):
    """Bar plot of average price by number of stops."""
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_price = df.groupby(stops_col)[target_col].mean().sort_values()
    sns.barplot(x=avg_price.index, y=avg_price.values, palette='Blues_d', ax=ax)
    ax.set_title('Average Price by Number of Stops')
    ax.set_xlabel('Number of Stops')
    ax.set_ylabel('Average Price (INR)')
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Seaborn heatmap of feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """Bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances (XGBoost)')
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_test, y_pred, model_name='XGBoost'):
    """Scatter plot of actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=20, label='Predictions')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    ax.set_xlabel('Actual Price (INR)')
    ax.set_ylabel('Predicted Price (INR)')
    ax.set_title(f'Actual vs Predicted ({model_name})')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_test, y_pred, model_name='XGBoost'):
    """Residual plot and residual distribution."""
    residuals = np.array(y_test) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color='darkorange', s=20)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Price')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residual Plot ({model_name})')
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color='mediumseagreen')
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_title('Residual Distribution')
    plt.tight_layout()
    return fig


def plot_monthly_trend(df, month_col='Journey_Month', target_col='Price'):
    """Line plot of average price per month."""
    if month_col not in df.columns:
        return None
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    monthly = df.groupby(month_col)[target_col].mean().reset_index()
    monthly[month_col] = monthly[month_col].map(month_map)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x=month_col, y=target_col,
                 marker='o', color='royalblue', linewidth=2.5, ax=ax)
    ax.fill_between(monthly[month_col], monthly[target_col], alpha=0.15, color='royalblue')
    ax.set_title('Average Flight Price by Month')
    plt.tight_layout()
    return fig
