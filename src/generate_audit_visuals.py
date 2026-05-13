import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import our cleaning functions
from location_cleaning import apply_location_normalization, apply_log_transformation

def generate_visuals(data_path='../dataset.csv'):
    print("--- Task 3: Audit Visuals (For Manas) ---")
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}. Please run eda.py first to generate it.")
        return
        
    df = pd.read_csv(data_path)
    
    # Preprocess
    df = apply_location_normalization(df, col_name='Location')
    df = apply_log_transformation(df, col_name='Price')
    
    sns.set_theme(style="whitegrid")
    
    # 1. 'Before vs. After' distribution of prices (Normal vs. Log-scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.histplot(df['Price'], kde=True, ax=axes[0], color='salmon')
    axes[0].set_title('Original Price Distribution (High Variance)')
    axes[0].set_xlabel('Price')
    
    sns.histplot(df['Log_Price'], kde=True, ax=axes[1], color='teal')
    axes[1].set_title('Log-Scaled Price Distribution (np.log1p)')
    axes[1].set_xlabel('Log_Price')
    
    plt.tight_layout()
    plt.savefig('../price_distribution_comparison.png', dpi=300)
    plt.close()
    print("1. Saved 'price_distribution_comparison.png'")
    
    # 2. 'Price vs. Location' bar chart for the top 10 most expensive areas
    # Aggregate to find the most expensive areas (by mean price)
    location_prices = df.groupby('Cleaned_Location')['Price'].mean().reset_index()
    top_locations = location_prices.sort_values(by='Price', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Price', y='Cleaned_Location', data=top_locations, palette='magma')
    plt.title('Top 10 Most Expensive Areas by Average Price')
    plt.xlabel('Average Price')
    plt.ylabel('Location')
    plt.tight_layout()
    plt.savefig('../price_vs_location_top10.png', dpi=300)
    plt.close()
    print("2. Saved 'price_vs_location_top10.png'")
    
    # 3. Outlier detection plot (Boxplot for continuous features)
    plt.figure(figsize=(12, 6))
    # We will look at Square_Feet and Age_Years for outliers
    # Standardize them just for visualization purposes so they fit on one plot, or plot separately
    # Let's plot Log_Price to see if outliers remain after transformation
    sns.boxplot(x=df['Log_Price'], color='lightblue')
    plt.title('Outlier Detection: Log_Price Boxplot')
    plt.xlabel('Log_Price')
    plt.tight_layout()
    plt.savefig('../outlier_detection_boxplot.png', dpi=300)
    plt.close()
    print("3. Saved 'outlier_detection_boxplot.png'")

if __name__ == "__main__":
    # If run from src/ directory, look one level up for house_prices_bangalore.csv
    generate_visuals(data_path=os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
