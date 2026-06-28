import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
import re
from data_utils import load_and_filter_data

DATA_PATH = "data/wait_times_join_attractions_themeparks_table_data-1778551975724.csv"
OUTPUT_DIR = Path("trained_models/xgboost_global")
us_holidays = holidays.US()

def get_safe_name(name):
    """Sanitizes strings for safe cross-platform file serialization."""
    if not name: return ""
    safe = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    while '__' in safe: safe = safe.replace('__', '_')
    return safe

def create_features(df):
    """
    Engineers discrete and continuous temporal features.
    Cyclical sin/cos transformations map the continuous nature of time 
    for the tree splits.
    """
    df = df.copy()
    df['hour'] = df['ds'].dt.hour
    df['minute'] = df['ds'].dt.minute
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['is_holiday'] = df['ds'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)
    df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    return df

def main():
    df = load_and_filter_data(DATA_PATH)
    resampled_dfs = []
    
    for park in df['tp_name'].unique():
        park_df = df[df['tp_name'] == park]
        for ride in park_df['at_name'].unique():
            ride_df = park_df[park_df['at_name'] == ride].copy()
            ride_df = ride_df.set_index('wait_time_upd_dt')
            
            # 30-minute interval resampling to establish a consistent temporal grid
            ride_df = ride_df['wait_time'].resample('30min').mean().dropna().reset_index()
            
            # Discard entities lacking sufficient statistical mass
            if len(ride_df) < 50: continue
                
            ride_df.rename(columns={'wait_time_upd_dt': 'ds', 'wait_time': 'y'}, inplace=True)
            ride_df['tp_name'] = park
            ride_df['at_name'] = get_safe_name(ride)
            resampled_dfs.append(ride_df)
            
    # Assemble the unified global dataframe
    global_df = pd.concat(resampled_dfs, ignore_index=True)
    global_df = global_df.sort_values('ds').reset_index(drop=True)
    global_df = create_features(global_df)
    
    # Snapshot the categorical ontology to ensure consistent encoding during inference
    park_categories = list(global_df['tp_name'].unique())
    ride_categories = list(global_df['at_name'].unique())
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "park_categories.json", 'w') as f: json.dump(park_categories, f)
    with open(OUTPUT_DIR / "ride_categories.json", 'w') as f: json.dump(ride_categories, f)
    
    # Cast identifiers to categorical dtype for native XGBoost handling
    global_df['tp_name'] = global_df['tp_name'].astype(pd.CategoricalDtype(categories=park_categories))
    global_df['at_name'] = global_df['at_name'].astype(pd.CategoricalDtype(categories=ride_categories))
    
    # Chronological dataset partitioning
    split_idx = int(len(global_df) * 0.8)
    train_df = global_df.iloc[:split_idx]
    
    features = [
        'tp_name', 'at_name', 'hour', 'minute', 'dayofweek', 'month', 
        'is_holiday', 'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    X_train, y_train = train_df[features], train_df['y']
    
    # Initialize the global model leveraging histogram-based tree building 
    # for accelerated processing of the large unified dataset.
    model = xgb.XGBRegressor(
        enable_categorical=True, 
        tree_method='hist', 
        max_depth=10, 
        n_estimators=300, 
        learning_rate=0.05, 
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    model.save_model(str(OUTPUT_DIR / "global_model.json"))

if __name__ == "__main__":
    main()
