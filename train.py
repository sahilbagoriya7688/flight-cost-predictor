"""
Main Training Script - Flight Cost Predictor
Run this script to preprocess data, train models, and generate EDA visualizations.

Usage:
    python train.py --data data/flight_price.csv
    python train.py --data data/flight_price.csv --no-tune   # skip GridSearch
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from data_preprocessing import preprocess_pipeline
from model_training import run_training_pipeline
from eda_visualization import (
    plot_price_distribution,
    plot_airline_vs_price,
    plot_stops_vs_price,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_monthly_trend,
)


def save_figure(fig, path: str):
    """Save matplotlib figure to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def run_eda(raw_df: pd.DataFrame, processed_df: pd.DataFrame, output_dir: str = "plots/"):
    """Generate and save EDA plots."""
    print("\n[EDA] Generating visualizations...")

    fig = plot_price_distribution(raw_df)
    save_figure(fig, f"{output_dir}price_distribution.png")

    if 'Airline' in raw_df.columns:
        fig = plot_airline_vs_price(raw_df)
        save_figure(fig, f"{output_dir}airline_vs_price.png")

    if 'Total_Stops' in raw_df.columns:
        fig = plot_stops_vs_price(raw_df)
        save_figure(fig, f"{output_dir}stops_vs_price.png")

    fig = plot_correlation_heatmap(processed_df)
    save_figure(fig, f"{output_dir}correlation_heatmap.png")

    fig = plot_monthly_trend(processed_df)
    if fig is not None:
        save_figure(fig, f"{output_dir}monthly_trend.png")

    print(f"[EDA] All plots saved to '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Flight Cost Predictor - Training Script")
    parser.add_argument("--data", type=str, default="data/flight_price.csv",
                        help="Path to raw CSV dataset")
    parser.add_argument("--model-dir", type=str, default="models/",
                        help="Directory to save trained models")
    parser.add_argument("--plot-dir", type=str, default="plots/",
                        help="Directory to save EDA plots")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip GridSearchCV (use default XGBoost params)")
    args = parser.parse_args()

    # Validate data path
    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset not found at '{args.data}'")
        print("Download from: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh/")
        sys.exit(1)

    # ── Step 1: Preprocessing ─────────────────────────────────────────
    print(f"\n[1/3] Loading and preprocessing: {args.data}")
    raw_df = pd.read_csv(args.data)
    print(f"      Raw shape: {raw_df.shape}")

    processed_df, encoders = preprocess_pipeline(args.data)
    print(f"      Processed shape: {processed_df.shape}")
    print(f"      Features: {list(processed_df.columns)}")

    # ── Step 2: EDA ───────────────────────────────────────────────────
    print("\n[2/3] Running EDA...")
    run_eda(raw_df, processed_df, output_dir=args.plot_dir)

    # ── Step 3: Training ──────────────────────────────────────────────
    print("\n[3/3] Training models...")
    best_model, all_metrics, X_test, y_test = run_training_pipeline(
        processed_df,
        target_col='Price',
        save_path=args.model_dir
    )

    # Save post-training plots
    y_pred = best_model.predict(X_test)
    feature_names = list(X_test.columns)

    fig = plot_actual_vs_predicted(y_test, y_pred)
    save_figure(fig, f"{args.plot_dir}actual_vs_predicted.png")

    fig = plot_residuals(y_test, y_pred)
    save_figure(fig, f"{args.plot_dir}residuals.png")

    if hasattr(best_model, 'feature_importances_'):
        fig = plot_feature_importance(best_model, feature_names)
        save_figure(fig, f"{args.plot_dir}feature_importance.png")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(" TRAINING COMPLETE")
    print("=" * 50)
    for model_name, metrics in all_metrics.items():
        print(f"  {model_name}:")
        print(f"    R2={metrics['R2 Score']}  MAE=₹{metrics['MAE']}  RMSE=₹{metrics['RMSE']}")
    print(f"\n  Models saved to: {args.model_dir}")
    print(f"  Plots saved to:  {args.plot_dir}")
    print("\nTo launch the web app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
