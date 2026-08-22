import pandas as pd

# Define valid operating windows to filter out overnight closures and 
# periods of missing data (e.g., Six Flags before Feb 15).
# Using a 24-hour format: [open_hour, close_hour)
PARK_CONSTRAINTS = {
    'Disneyland': {'start_date': '2025-12-06', 'open_hour': 8, 'close_hour': 24}, 
    'Disney California Adventure Park': {'start_date': '2025-12-06', 'open_hour': 8, 'close_hour': 22},
    'Universal Studios Hollywood': {'start_date': '2025-12-06', 'open_hour': 8, 'close_hour': 22}, 
    'SeaWorld San Diego': {'start_date': '2025-12-06', 'open_hour': 10, 'close_hour': 20},
    'SeaWorld San Diego Obsolete': {'start_date': '2025-12-06', 'open_hour': 10, 'close_hour': 20},
    'Six Flags Magic Mountain': {'start_date': '2026-02-15', 'open_hour': 10, 'close_hour': 21}
}

def load_and_filter_data(data_path):
    """
    Loads wait time data and applies domain-specific filters.
    Retains valid 0-minute wait times (walk-ons) while discarding 
    artificial 0s generated during park closures or scraping outages.
    """
    df = pd.read_csv(data_path, usecols=['wait_time_upd_dt', 'tp_name', 'at_name', 'wait_time'])
    df['wait_time_upd_dt'] = pd.to_datetime(df['wait_time_upd_dt'])
    
    # Extract temporal component for heuristic filtering
    df['hour'] = df['wait_time_upd_dt'].dt.hour
    
    # Allow 0s to capture true walk-on states, but bound the upper limit 
    # to 900 minutes to drop extreme outliers/sensor errors.
    df = df[(df['wait_time'] >= 0) & (df['wait_time'] < 900)]
    
    valid_rows = []
    for park, constraints in PARK_CONSTRAINTS.items():
        park_df = df[df['tp_name'] == park]
        
        # Isolate data within the park's valid operational timeframe
        park_df = park_df[
            (park_df['wait_time_upd_dt'] >= constraints['start_date']) &
            (park_df['hour'] >= constraints['open_hour']) &
            (park_df['hour'] < constraints['close_hour'])
        ]
        valid_rows.append(park_df)
        
    filtered_df = pd.concat(valid_rows, ignore_index=True)
    return filtered_df.drop(columns=['hour'])
