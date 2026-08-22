import json
from pathlib import Path
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error
import re
from data_utils import load_and_filter_data

DATA_PATH = "data/wait_times_join_attractions_themeparks_table_data-1778551975724.csv"
OUTPUT_DIR = Path("trained_models/prophet")

def main():
    df = load_and_filter_data(DATA_PATH)
    parks = df['tp_name'].unique()
    results = []
    
    for park in parks:
        park_df = df[df['tp_name'] == park]
        rides = park_df['at_name'].unique()
        
        park_dir = OUTPUT_DIR / park.replace(' ', '_').replace('/', '_')
        park_dir.mkdir(parents=True, exist_ok=True)
        
        for ride in rides:
            ride_df = park_df[park_df['at_name'] == ride].copy()
            
            # Resample to 30-min intervals to smooth micro-fluctuations 
            # and prevent over-indexing on high-frequency noise.
            ride_df = ride_df.set_index('wait_time_upd_dt')
            ride_df = ride_df['wait_time'].resample('30min').mean().dropna().reset_index()
            ride_df.rename(columns={'wait_time_upd_dt': 'ds', 'wait_time': 'y'}, inplace=True)
            
            # Skip attractions with insufficient historical variance for Prophet
            if len(ride_df) < 50:
                continue
                
            # Chronological split to prevent temporal data leakage
            split_idx = int(len(ride_df) * 0.8)
            train_df = ride_df.iloc[:split_idx]
            test_df = ride_df.iloc[split_idx:]
            
            # Prophet configuration: assuming bounded growth and standard seasonalities.
            # Yearly seasonality is disabled due to the dataset's limited time horizon (< 1 year).
            model = Prophet(growth='flat', daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            model.add_country_holidays(country_name='US')
            model.fit(train_df)
            
            forecast = model.predict(test_df[['ds']])
            mae = mean_absolute_error(test_df['y'], forecast['yhat'])
            
            # Sanitize model artifacts for cross-platform filesystem compatibility
            safe_ride = re.sub(r'[^\w\s-]', '', ride).strip().replace(' ', '_').replace('__', '_')
            model_path = park_dir / f"{safe_ride}_model.json"
            
            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_to_json(model), f)
                
            results.append({
                'park': park,
                'ride': ride,
                'mae': round(mae, 2),
                'train_samples': len(train_df),
                'test_samples': len(test_df)
            })
            
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "training_summary.csv", index=False)

if __name__ == "__main__":
    main()
