import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib
import os

def run_xgboost_model(data_path='../house_prices_bangalore.csv'):
    print("===================================================")
    print("PHASE 2: XGBOOST MODEL (RMSLE FOCUSED)")
    print("===================================================")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"\nOriginal dataset shape: {df.shape}")
    print(df.columns.tolist())

    # 2. Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"After cleaning: {df.shape}")

    # 3. Advanced Feature Engineering
    # Using all 12 features: area, location, bhk, bath, balcony, parking, furnishing, property_type, age, total_rooms, area_per_bhk, bath_to_bhk_ratio
    df['total_rooms']       = df['bhk'] + df['bath']
    df['area_per_bhk']      = df['area'] / df['bhk'].replace(0, 1)
    df['bath_to_bhk_ratio'] = df['bath'] / df['bhk'].replace(0, 1)
    df['is_new_property']   = (df['age'] <= 5).astype(int)

    # 4. Outlier Removal using IQR on price
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
    print(f"After outlier removal: {df.shape}")

    # 5. Apply np.log1p() to price for RMSLE optimization
    # log(price+1) compresses the huge price variance, reducing RMSLE significantly
    df['log_price'] = np.log1p(df['price'])

    # 6. Encode Categorical Features
    # XGBoost requires numeric input, so we use Label Encoding for categoricals
    le = LabelEncoder()
    for col in ['location', 'furnishing', 'property_type']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 7. Drop irrelevant/noisy cols, keep 12 features
    feature_cols = ['area', 'location', 'bhk', 'bath', 'balcony', 'parking',
                    'furnishing', 'property_type', 'age',
                    'total_rooms', 'area_per_bhk', 'bath_to_bhk_ratio']
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df['log_price']   # Train on log_price → RMSE of log = RMSLE of price

    # 8. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 9. XGBoost Model with tuned hyperparameters
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 10. Evaluate
    predictions = xgb.predict(X_test)
    rmsle = np.sqrt(mean_squared_error(y_test, predictions))
    r2    = r2_score(y_test, predictions)

    print("\n================ XGBOOST MODEL METRICS ================")
    print(f"R² Score        : {r2:.4f}  ({r2*100:.2f}%)")
    print(f"RMSLE           : {rmsle:.4f}")
    
    # 11. Feature Importance Plot
    plt.figure(figsize=(12, 6))
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': xgb.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
    plt.title('XGBoost - Top Feature Importances')
    plt.tight_layout()
    plt.savefig('../xgboost_feature_importance.png', dpi=300)
    plt.close()
    print("\nSaved: xgboost_feature_importance.png")

    # 12. Save final model
    joblib.dump(xgb, '../xgboost_model.pkl')
    print("Saved: xgboost_model.pkl")

    # 13. Write final_metrics.txt
    metrics_path = '../final_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("====================================\n")
        f.write("  FINAL MODEL METRICS - DataVerse\n")
        f.write("  Track 2: Bengaluru House Prices\n")
        f.write("====================================\n\n")
        f.write(f"Model           : XGBoost Regressor\n")
        f.write(f"Features Used   : {len(feature_cols)} features\n")
        f.write(f"Feature List    : {feature_cols}\n\n")
        f.write(f"R² Score        : {r2:.4f} ({r2*100:.2f}%)\n")
        f.write(f"RMSLE           : {rmsle:.4f}\n\n")
        f.write("Notes:\n")
        f.write("- Target variable 'price' transformed using np.log1p() for RMSLE optimization\n")
        f.write("- IQR-based outlier removal applied to remove luxury price anomalies\n")
        f.write("- Advanced feature engineering: total_rooms, area_per_bhk, bath_to_bhk_ratio\n")
    print(f"Saved: final_metrics.txt")
    return r2, rmsle

if __name__ == "__main__":
    run_xgboost_model(os.path.join(os.path.dirname(__file__), '..', 'house_prices_bangalore.csv'))
