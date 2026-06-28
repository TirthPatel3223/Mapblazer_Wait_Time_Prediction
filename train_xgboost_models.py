from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import holidays
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import re
from data_utils import load_and_filter_data

DATA_PATH = "data/wait_times_join_attractions_themeparks_table_data-1778551975724.csv"
OUTPUT_DIR = Path("trained_models/xgboost")
us_holidays = holidays.US()

def create_features(df):
    """
    Engineers discrete and continuous temporal features.
    Cyclical features (sin/cos transformations) are crucial here to help the tree 
    understand the continuous boundary between periods (e.g., Dec 31 to Jan 1).
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
    parks = df['tp_name'].unique()
    results = []
    
    # Coarse grid for fast hyperparameter tuning. 
    # Balances exploration space with computational constraints.
    param_distributions = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    for park in parks:
        park_df = df[df['tp_name'] == park]
        rides = park_df['at_name'].unique()
        
        park_dir = OUTPUT_DIR / park.replace(' ', '_').replace('/', '_')
        park_dir.mkdir(parents=True, exist_ok=True)
        
        for ride in rides:
            ride_df = park_df[park_df['at_name'] == ride].copy()
            
            # 30-minute interval resampling to aggregate micro-variance
            ride_df = ride_df.set_index('wait_time_upd_dt')
            ride_df = ride_df['wait_time'].resample('30min').mean().dropna().reset_index()
            ride_df.rename(columns={'wait_time_upd_dt': 'ds', 'wait_time': 'y'}, inplace=True)
            
            if len(ride_df) < 50:
                continue
                
            # Chronological split preserves temporal integrity during validation
            split_idx = int(len(ride_df) * 0.8)
            train_df = create_features(ride_df.iloc[:split_idx])
            test_df = create_features(ride_df.iloc[split_idx:])
            
            features = [
                'hour', 'minute', 'dayofweek', 'month', 'is_holiday', 
                'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
            ]
            X_train, y_train = train_df[features], train_df['y']
            X_test, y_test = test_df[features], test_df['y']
            
            # Base estimator setup; leveraging multi-threading for faster tree construction
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model, param_distributions, n_iter=10, 
                scoring='neg_mean_absolute_error', cv=3, random_state=42, n_jobs=-1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            forecast = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, forecast)
            
            safe_ride = re.sub(r'[^\w\s-]', '', ride).strip().replace(' ', '_').replace('__', '_')
            best_model.save_model(str(park_dir / f"{safe_ride}_model.json"))
                
            results.append({
                'park': park, 'ride': ride, 'mae': round(mae, 2), 
                'best_params': str(search.best_params_), 
                'train_samples': len(train_df), 'test_samples': len(test_df)
            })
            
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "training_summary.csv", index=False)

if __name__ == "__main__":
    main()
