import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys

# Import our cleaning functions
from location_cleaning import apply_location_normalization, apply_log_transformation

def run_optimized_rf(data_path='../house_prices_bangalore.csv'):
    print("\n===================================================")
    print("PHASE 2: OPTIMIZED ENSEMBLE MODEL (RMSLE FOCUSED)")
    print("===================================================\n")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Could not find dataset at {data_path}.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")
    
    # 2. Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Drop weak features to reduce noise
    df = df.drop(['balcony', 'parking'], axis=1, errors='ignore')
    
    # 3. Feature Engineering
    df['total_rooms'] = df['bhk'] + df['bath']
    
    # 4. Outlier Removal (Using IQR on Price)
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df = df[(df['price'] >= lower_limit) & (df['price'] <= upper_limit)]
    print(f"Dataset shape after outlier removal: {df.shape}")
    
    # 5. Apply Team Integration Tools
    # Location normalization (handles high variance in names)
    df = apply_location_normalization(df, col_name='location')
    
    # Apply Log Transformation for RMSLE metric target
    df = apply_log_transformation(df, col_name='price')
    
    # 6. Prepare Features and Target
    # One-hot encode categorical features
    categorical_cols = ['Cleaned_Location', 'furnishing', 'property_type']
    # Ensure they exist before encoding
    cols_to_encode = [c for c in categorical_cols if c in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    
    # Drop original non-numeric columns
    features = df_encoded.drop(['location', 'price', 'Log_price'], axis=1, errors='ignore')
    target = df_encoded['Log_price'] # Train on Log-Scaled Price for RMSLE
    
    # 7. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # 8. Hyperparameter Tuning using RandomizedSearchCV
    print("\nTuning Hyperparameters for Gradient Boosting...")
    rf = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                   n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    
    best_rf = rf_random.best_estimator_
    print(f"Best Parameters Found: {rf_random.best_params_}")
    
    # 9. Evaluate
    predictions = best_rf.predict(X_test)
    
    # Since we predicted Log_Price, the RMSE of these predictions IS the RMSLE of the actual prices!
    rmsle = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n================ MODEL METRICS ================")
    print(f"Model Accuracy (R² Score): {r2 * 100:.2f}%")
    print(f"RMSLE (Track 2 Objective): {rmsle:.4f}")
    
    # 10. Feature Importance Plot
    importances = best_rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
    plt.title('Top 10 Feature Importances - Optimized Random Forest')
    plt.tight_layout()
    plot_path = '../feature_importance_rf.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # 11. Save Model
    joblib.dump(best_rf, '../optimized_rf_model.pkl')
    print(f"\nSaved tuned model to optimized_rf_model.pkl")

if __name__ == "__main__":
    run_optimized_rf(data_path=os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
