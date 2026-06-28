# Theme Park Wait Time Predictor

A comprehensive Machine Learning pipeline designed to predict theme park ride wait times using a mix of historical datasets and real-time telemetry from the Queue-Times API. 

This repository leverages three different architectures—Time Series Analysis, Localized Gradient Boosting, and a Unified Global Gradient Boosting approach—to analyze temporal patterns, holidays, and park-specific constraints.

## Features
*   **Three Distinct ML Architectures:**
    *   **Prophet (Time Series):** Utilizes Facebook Prophet for robust chronological time-series forecasting.
    *   **XGBoost Local:** Trains a highly specialized, independent Gradient Boosted Tree for every individual ride.
    *   **XGBoost Global:** Trains a massive, unified Gradient Boosted Tree across all parks and rides, utilizing strict categorical embeddings to map relationships across the dataset.
*   **Advanced Feature Engineering:** Automatically encodes continuous temporal data into cyclical sine/cosine features (e.g., smoothly mapping the transition from 23:59 to 00:00).
*   **Robust Operational Filtering:** Intelligently filters out erroneous "zero" wait times during park closures while preserving true "walk-on" zeros during operating hours.
*   **Real-time API Inference:** Includes a live testing harness that pings the Queue-Times API, fetches the current wait times, runs inference against all 3 model architectures simultaneously, and compares them against the historical baseline average.

## Project Structure
```text
├── trained_models/
│   ├── prophet/            # Serialized Prophet JSON models
│   ├── xgboost/            # Serialized XGBoost Local JSON models
│   └── xgboost_global/     # Global model JSON + Categorical Ontologies
├── data_utils.py                    # Centralized ETL and filtering logic
├── train_prophet_models.py          # Training pipeline for Prophet
├── train_xgboost_models.py          # Training pipeline for XGBoost (Local)
├── train_xgboost_global.py          # Training pipeline for XGBoost (Global)
├── real_time_testing/
│   ├── test_realtime_predictions.py         # Live API inference & evaluation script
│   └── plot_error_histograms.py             # Visualizes Mean Absolute Error distributions
└── model_testing_scripts/
    ├── evaluate_test_data.py                # Generates holdout test data histograms
    └── evaluate_high_traffic_test_data.py   # Generates weekend/holiday data histograms
```

## Setup & Installation

**Prerequisites:** Python 3.9+

Install the required dependencies:
```bash
pip install pandas numpy xgboost prophet requests matplotlib seaborn holidays
```

## Training the Models
The training dataset must be located at `data/wait_times_join_attractions_themeparks_table_data-1778551975724.csv`.

To completely rebuild the model artifacts from the raw data, execute the training scripts (they automatically utilize multiprocessing to speed up training):
```bash
python train_xgboost_global.py
python train_xgboost_models.py
python train_prophet_models.py
```
*Note: The global model is significantly faster to train than the localized loops.*

## Live Real-time Inference
Once the models are populated inside `trained_models/`, you can query the live API to test how well the models are performing at this exact second:
```bash
python real_time_testing/test_realtime_predictions.py
```
This script will output a tabular sample to your console and serialize the full results into `data/realtime_comparison_v2.csv`.

## Error Analysis

To rigorously evaluate the models on the historical 20% holdout testing dataset:
```bash
python model_testing_scripts/evaluate_test_data.py
python model_testing_scripts/evaluate_high_traffic_test_data.py
```

To visualize the spread and the Mean Absolute Error (MAE) of your live real-time predictions across the different architectures:
```bash
python real_time_testing/plot_error_histograms.py
```
This will output an `error_histograms.png` graphic containing a 2x2 grid of KDE distributions plotting `Predicted Wait - Actual Wait`.

---
*Disclaimer: The testing of the model in real time for visulation relies on the Queue-Times API. The data used to train the models was provided by Mapblazer Team.*