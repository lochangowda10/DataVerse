import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os
import sys

# Import our cleaning functions
from location_cleaning import apply_location_normalization, apply_log_transformation

def run_baseline_model(data_path='../dataset.csv'):
    print("--- Task 2: Baseline Random Forest Regressor ---")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}. Please run eda.py first to generate it.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Data Cleaning & Preprocessing (Hanamanreddy's tasks)
    # Apply location normalization
    df = apply_location_normalization(df, col_name='Location')
    
    # Apply Log Transformation for RMSLE metric
    df = apply_log_transformation(df, col_name='Price')
    
    # Handle Missing Values
    # Impute Square_Feet with median
    sqft_imputer = SimpleImputer(strategy='median')
    df['Square_Feet'] = sqft_imputer.fit_transform(df[['Square_Feet']])
    
    # Impute Age_Years with median
    age_imputer = SimpleImputer(strategy='median')
    df['Age_Years'] = age_imputer.fit_transform(df[['Age_Years']])
    
    # 3. Feature Engineering
    # We must one-hot encode the 'Cleaned_Location' for the Random Forest
    df_encoded = pd.get_dummies(df, columns=['Cleaned_Location'], drop_first=True)
    
    # Select features and target
    # We drop the original 'Location' and 'Price' (since we predict Log_Price)
    features = df_encoded.drop(['Location', 'Price', 'Log_Price'], axis=1)
    target = df_encoded['Log_Price']
    
    # 4. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    # 5. Model Training (Baseline Random Forest)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Baseline Validation RMSLE (since target is log-scaled): {rmse:.4f}")
    
    # 6. Feature Importance Plot
    importances = rf_model.feature_importances_
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
    plt.title('Top 10 Feature Importances - Random Forest Regressor')
    plt.xlabel('Importance (Reduction in Impurity)')
    plt.ylabel('Features')
    plt.tight_layout()
    
    # Save the plot for the presentation
    plot_path = '../feature_importance.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")

if __name__ == "__main__":
    # If run from src/ directory, look one level up for house_prices_bangalore.csv
    run_baseline_model(data_path=os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
