import os
import re
import json
import logging
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import holidays
from datetime import datetime, timezone
from pathlib import Path
from prophet.serialize import model_from_json
from typing import Dict, Any, Optional, Tuple

# Configure logging for production-grade visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants & Ontological Mappings
PARK_MAPPING = {
    16: "Disneyland",
    17: "Disney California Adventure Park",
    32: "Six Flags Magic Mountain",
    66: "Universal Studios Hollywood",
    20: "SeaWorld San Diego",
}

API_BASE_URL = "https://queue-times.com/parks/{park_id}/queue_times.json"
MODEL_DIR_PROPHET = Path("trained_models/prophet")
MODEL_DIR_XGB_LOCAL = Path("trained_models/xgboost")
MODEL_DIR_XGB_GLOBAL = Path("trained_models/xgboost_global")

US_HOLIDAYS = holidays.US()

def sanitize_filename(name: str) -> str:
    """
    Sanitizes entity names to strictly match the string serialization convention 
    applied during model training artifacts creation.
    """
    if not name:
        return ""
    safe = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe

def fetch_live_queue_data() -> Dict[int, Any]:
    """
    Fetches real-time queue telemetry from the Queue-Times API for all mapped theme parks.
    """
    current_utc = datetime.now(timezone.utc)
    logging.info(f"Fetching live telemetry for {current_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC...")
    
    live_data = {}
    for park_id, park_name in PARK_MAPPING.items():
        url = API_BASE_URL.format(park_id=park_id)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            live_data[park_id] = resp.json()
            logging.info(f"Successfully fetched stream for {park_name}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch telemetry for {park_name} (ID: {park_id}): {e}")
            
    return live_data

def build_temporal_features(dt: datetime) -> Dict[str, float]:
    """
    Extracts base and cyclical temporal features required by the XGBoost trees.
    Cyclical features map the continuous nature of time (e.g. 23:59 to 00:00).
    """
    return {
        'hour': dt.hour,
        'minute': dt.minute,
        'dayofweek': dt.dayofweek,
        'month': dt.month,
        'is_holiday': 1 if dt.date() in US_HOLIDAYS else 0,
        'is_weekend': 1 if dt.dayofweek in [5, 6] else 0,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24.0),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24.0),
        'month_sin': np.sin(2 * np.pi * dt.month / 12.0),
        'month_cos': np.cos(2 * np.pi * dt.month / 12.0)
    }

def load_global_xgboost() -> Tuple[Optional[xgb.XGBRegressor], Optional[pd.CategoricalDtype], Optional[pd.CategoricalDtype]]:
    """
    Loads the global XGBoost model and strictly enforces the categorical ontology 
    (park and ride names) defined during the training phase.
    """
    global_model_path = MODEL_DIR_XGB_GLOBAL / "global_model.json"
    if not global_model_path.exists():
        logging.warning("Global XGBoost model not found. Proceeding without global predictions.")
        return None, None, None
        
    try:
        g_model = xgb.XGBRegressor(enable_categorical=True)
        g_model.load_model(str(global_model_path))
        
        with open(MODEL_DIR_XGB_GLOBAL / "park_categories.json", 'r') as f:
            park_cats = json.load(f)
        with open(MODEL_DIR_XGB_GLOBAL / "ride_categories.json", 'r') as f:
            ride_cats = json.load(f)
            
        cat_type_park = pd.CategoricalDtype(categories=park_cats)
        cat_type_ride = pd.CategoricalDtype(categories=ride_cats)
        return g_model, cat_type_park, cat_type_ride
    except Exception as e:
        logging.error(f"Error loading global model or categorical mappings: {e}")
        return None, None, None

def evaluate_realtime_predictions():
    """
    Core execution pipeline: 
    1. Fetches live telemetry.
    2. Extracts and encodes temporal features.
    3. Runs inference across all three model architectures (Prophet, XGBoost Local, XGBoost Global).
    4. Compiles comparative error metrics.
    """
    live_data = fetch_live_queue_data()
    if not live_data:
        logging.error("No live data available to process.")
        return
        
    g_model, cat_park, cat_ride = load_global_xgboost()
    results = []
    
    for park_id, data in live_data.items():
        park_name = PARK_MAPPING[park_id]
        
        p_dir = MODEL_DIR_PROPHET / park_name.replace(' ', '_').replace('/', '_')
        x_dir = MODEL_DIR_XGB_LOCAL / park_name.replace(' ', '_').replace('/', '_')
        
        # Flatten ride hierarchy across potential theme park 'lands'
        rides = data.get("rides", [])
        for land in data.get("lands", []):
            rides.extend(land.get("rides", []))
            
        for ride in rides:
            if not ride.get("is_open") or ride.get("last_updated") is None:
                continue
                
            actual_wait = ride.get("wait_time", 0)
            ride_name = ride.get("name", "")
            
            try:
                # Strip timezone awareness to mirror training pipeline schemas
                dt = pd.to_datetime(ride.get("last_updated")).tz_localize(None)
            except Exception:
                continue
                
            safe_ride_name = sanitize_filename(ride_name)
            p_model_path = p_dir / f"{safe_ride_name}_model.json"
            x_model_path = x_dir / f"{safe_ride_name}_model.json"
            
            p_pred, baseline_pred, x_pred, g_pred = None, None, None, None
            
            # --- 1. Prophet Inference ---
            if p_model_path.exists():
                try:
                    with open(p_model_path, 'r', encoding='utf-8') as f:
                        p_model = model_from_json(json.load(f))
                    forecast = p_model.predict(pd.DataFrame({'ds': [dt]}))
                    p_pred = max(0, round(forecast['yhat'].iloc[0]))
                    baseline_pred = round(p_model.history['y'].mean())
                except Exception:
                    pass
                    
            # --- 2. XGBoost Local Inference ---
            temporal_features = build_temporal_features(dt)
            if x_model_path.exists():
                try:
                    x_model = xgb.XGBRegressor()
                    x_model.load_model(x_model_path)
                    x_features = pd.DataFrame([temporal_features])
                    x_pred = max(0, round(x_model.predict(x_features)[0]))
                except Exception:
                    pass
                    
            # --- 3. XGBoost Global Inference ---
            if g_model is not None:
                try:
                    g_features_dict = {
                        'tp_name': park_name,
                        'at_name': safe_ride_name,
                        **temporal_features
                    }
                    g_features = pd.DataFrame([g_features_dict])
                    
                    # Apply strict categorical dtypes learned during training
                    g_features['tp_name'] = g_features['tp_name'].astype(cat_park)
                    g_features['at_name'] = g_features['at_name'].astype(cat_ride)
                    
                    g_pred = max(0, round(g_model.predict(g_features)[0]))
                except Exception:
                    pass
                    
            # Persist record only if all models successfully executed inference
            if all(v is not None for v in [baseline_pred, p_pred, x_pred, g_pred]):
                results.append({
                    "Park": park_name,
                    "Ride": ride_name,
                    "Time": dt.strftime('%Y-%m-%d %H:%M'),
                    "Actual Wait": actual_wait,
                    "Baseline Predict": baseline_pred,
                    "Prophet Predict": p_pred,
                    "XGB Local Predict": x_pred,
                    "XGB Global Predict": g_pred,
                    "Prophet Error": abs(actual_wait - p_pred),
                    "XGB Local Error": abs(actual_wait - x_pred),
                    "XGB Global Error": abs(actual_wait - g_pred)
                })
                
    if results:
        results_df = pd.DataFrame(results)
        logging.info(f"Processed {len(results_df)} valid inferences across all architectures.")
        
        print("\n" + "="*80)
        print("REALTIME INFERENCE REPORT (SAMPLE)")
        print("="*80)
        print(results_df[['Park', 'Ride', 'Actual Wait', 'Baseline Predict', 'Prophet Predict', 'XGB Local Predict', 'XGB Global Predict']].head(20).to_string(index=False))
        
        out_path = "data/realtime_comparison_v2.csv"
        results_df.to_csv(out_path, index=False)
        logging.info(f"Full inference matrix serialized to '{out_path}'.")
    else:
        logging.warning("No overlapping matches found between open API rides and trained model schemas.")

if __name__ == "__main__":
    evaluate_realtime_predictions()
