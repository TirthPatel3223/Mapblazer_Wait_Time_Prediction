import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from prophet.serialize import model_from_json
import holidays
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import load_and_filter_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "data/wait_times_join_attractions_themeparks_table_data-1778551975724.csv"
MODEL_DIR_PROPHET = "trained_models/prophet"
MODEL_DIR_XGB_LOCAL = "trained_models/xgboost"
MODEL_DIR_XGB_GLOBAL = "trained_models/xgboost_global"

us_holidays = holidays.US()

def create_features(df):
    """Generates temporal features identically to the training pipeline."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_holiday'] = df.index.normalize().map(lambda x: 1 if x in us_holidays else 0)
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    return df

def main():
    logging.info(f"Loading and filtering historical data from {DATA_PATH}...")
    df = load_and_filter_data(DATA_PATH)
    
    # Load global model once
    g_model = None
    global_model_path = os.path.join(MODEL_DIR_XGB_GLOBAL, "global_model.json")
    if os.path.exists(global_model_path):
        g_model = xgb.XGBRegressor(enable_categorical=True)
        g_model.load_model(global_model_path)
        with open(os.path.join(MODEL_DIR_XGB_GLOBAL, "park_categories.json"), 'r') as f:
            cat_type_park = pd.CategoricalDtype(categories=json.load(f))
        with open(os.path.join(MODEL_DIR_XGB_GLOBAL, "ride_categories.json"), 'r') as f:
            cat_type_ride = pd.CategoricalDtype(categories=json.load(f))
            
    results = []
    
    for park in df['tp_name'].unique():
        park_df = df[df['tp_name'] == park]
        park_folder = park.replace(' ', '_').replace('/', '_')
        
        for ride in park_df['at_name'].unique():
            ride_df = park_df[park_df['at_name'] == ride].copy()
            ride_df = ride_df.set_index('wait_time_upd_dt')
            ride_df = ride_df[['wait_time']].resample('30min').mean().dropna()
            
            # Need minimum data to have a valid train/test split
            if len(ride_df) < 100:
                continue
                
            # Chronological 80/20 split (matching the training script exactly)
            split_idx = int(len(ride_df) * 0.8)
            train_df = ride_df.iloc[:split_idx]
            test_df = ride_df.iloc[split_idx:].copy()
            
            baseline_val = train_df['wait_time'].mean()
            safe_ride = ride.replace(' ', '_').replace('/', '_')
            
            p_model_path = os.path.join(MODEL_DIR_PROPHET, park_folder, f"{safe_ride}_model.json")
            x_model_path = os.path.join(MODEL_DIR_XGB_LOCAL, park_folder, f"{safe_ride}_model.json")
            
            # Create feature matrix early so we can filter by weekends/holidays
            x_features = create_features(test_df)
            
            # 1. Predict Prophet
            if os.path.exists(p_model_path):
                try:
                    with open(p_model_path, 'r') as f:
                        p_model = model_from_json(json.load(f))
                    future = pd.DataFrame({'ds': test_df.index})
                    forecast = p_model.predict(future)
                    test_df['prophet_pred'] = np.clip(np.round(forecast['yhat'].values), 0, None)
                except Exception:
                    test_df['prophet_pred'] = np.nan
            else:
                test_df['prophet_pred'] = np.nan
                
            # 2. Predict XGBoost Local
            if os.path.exists(x_model_path):
                try:
                    x_model = xgb.XGBRegressor()
                    x_model.load_model(x_model_path)
                    features_cols = ['hour', 'minute', 'dayofweek', 'month', 'is_holiday', 'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
                    test_df['xgb_local_pred'] = np.clip(np.round(x_model.predict(x_features[features_cols])), 0, None)
                except Exception:
                    test_df['xgb_local_pred'] = np.nan
            else:
                test_df['xgb_local_pred'] = np.nan
                
            # 3. Predict XGBoost Global
            if g_model is not None:
                try:
                    g_features = x_features.copy()
                    g_features['tp_name'] = pd.Series(park, index=g_features.index, dtype=cat_type_park)
                    g_features['at_name'] = pd.Series(safe_ride, index=g_features.index, dtype=cat_type_ride)
                    features_cols = ['tp_name', 'at_name', 'hour', 'minute', 'dayofweek', 'month', 'is_holiday', 'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
                    test_df['xgb_global_pred'] = np.clip(np.round(g_model.predict(g_features[features_cols])), 0, None)
                except Exception:
                    test_df['xgb_global_pred'] = np.nan
            else:
                test_df['xgb_global_pred'] = np.nan
                
            # Collect valid results WHERE is_weekend OR is_holiday
            test_df['is_weekend'] = x_features['is_weekend']
            test_df['is_holiday'] = x_features['is_holiday']
            
            valid_mask = test_df[['prophet_pred', 'xgb_local_pred', 'xgb_global_pred']].notna().all(axis=1)
            high_traffic_mask = (test_df['is_weekend'] == 1) | (test_df['is_holiday'] == 1)
            
            for _, row in test_df[valid_mask & high_traffic_mask].iterrows():
                results.append({
                    "Actual Wait": row['wait_time'],
                    "Baseline Predict": baseline_val,
                    "Prophet Predict": row['prophet_pred'],
                    "XGB Local Predict": row['xgb_local_pred'],
                    "XGB Global Predict": row['xgb_global_pred']
                })
                    
    results_df = pd.DataFrame(results)
    logging.info(f"Generated predictions for {len(results_df)} high-traffic test samples across all rides.")
    
    if len(results_df) == 0:
        logging.error("No valid predictions were generated.")
        return
        
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("High Traffic (Weekends & Holidays) Prediction Errors", fontsize=18, fontweight='bold', y=0.98)
    axes = axes.flatten()
    
    models = {
        "Baseline (Historical Average)": "Baseline Predict",
        "Prophet (Time Series)": "Prophet Predict",
        "XGBoost Local (Individual Trees)": "XGB Local Predict",
        "XGBoost Global (Unified Tree)": "XGB Global Predict"
    }
    
    for i, (model_name, col_name) in enumerate(models.items()):
        error = results_df[col_name] - results_df["Actual Wait"]
        mae = np.abs(error).mean()
        rmse = np.sqrt((error ** 2).mean())
        
        ax = axes[i]
        sns.histplot(error, kde=True, ax=ax, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
        ax.set_title(f"{model_name}\nMAE = {mae:.2f} mins | RMSE = {rmse:.2f}", fontsize=14, pad=10)
        ax.set_xlabel("Error in Minutes (Predicted - Actual)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        
        ax.axvline(x=0, color='darkgreen', linestyle='-', linewidth=2, label='Perfect Prediction (0)')
        ax.axvline(x=mae, color='crimson', linestyle='--', linewidth=2, label=f'+ MAE ({mae:.2f})')
        ax.axvline(x=-mae, color='crimson', linestyle='--', linewidth=2, label=f'- MAE (-{mae:.2f})')
        
        ax.set_xlim(-120, 120)
        ax.legend(fontsize=10)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = "high_traffic_error_histograms.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"High traffic test data plots successfully saved to '{output_path}'.")

if __name__ == "__main__":
    main()
