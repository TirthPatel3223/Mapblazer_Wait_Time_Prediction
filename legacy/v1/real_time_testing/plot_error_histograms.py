import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_error_histograms(csv_path="data/realtime_comparison_v2.csv", output_path="error_histograms.png"):
    """
    Reads the realtime predictions CSV, computes the prediction errors, 
    and generates a 2x2 grid of histograms showing the error distribution 
    and Mean Absolute Error (MAE) for all four models.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Data file '{csv_path}' not found. Please run predictions first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Dictionary mapping display names to their respective dataframe columns
    models = {
        "Baseline (Historical Average)": "Baseline Predict",
        "Prophet (Time Series)": "Prophet Predict",
        "XGBoost Local (Individual Trees)": "XGB Local Predict",
        "XGBoost Global (Unified Tree)": "XGB Global Predict"
    }
    
    # Set up a clean, professional plotting style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Prediction Error Distributions (Predicted Wait - Actual Wait)", fontsize=18, fontweight='bold', y=0.98)
    axes = axes.flatten()
    
    for i, (model_name, col_name) in enumerate(models.items()):
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' missing. Skipping {model_name}.")
            continue
            
        # Calculate raw error (Positive = Overpredicted, Negative = Underpredicted)
        error = df[col_name] - df["Actual Wait"]
        
        # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
        mae = np.abs(error).mean()
        rmse = np.sqrt((error ** 2).mean())
        
        ax = axes[i]
        
        # Plot histogram with Kernel Density Estimate
        sns.histplot(error, kde=True, ax=ax, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Dynamic title with MAE and RMSE explicitly stated
        ax.set_title(f"{model_name}\nMAE = {mae:.2f} mins | RMSE = {rmse:.2f}", fontsize=14, pad=10)
        ax.set_xlabel("Error in Minutes (Predicted - Actual)", fontsize=12)
        ax.set_ylabel("Frequency (Number of Rides)", fontsize=12)
        
        # Add vertical line for Zero Error (Perfect Prediction)
        ax.axvline(x=0, color='darkgreen', linestyle='-', linewidth=2, label='Perfect Prediction (0)')
        
        # Add vertical lines representing the spread of the Mean Absolute Error
        ax.axvline(x=mae, color='crimson', linestyle='--', linewidth=2, label=f'+ MAE ({mae:.2f})')
        ax.axvline(x=-mae, color='crimson', linestyle='--', linewidth=2, label=f'- MAE (-{mae:.2f})')
        
        ax.legend(fontsize=10)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save high-resolution plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! High-resolution histograms saved to '{output_path}'")

if __name__ == "__main__":
    generate_error_histograms()
