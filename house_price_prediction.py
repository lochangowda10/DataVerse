# ================================
# HOUSE PRICE PREDICTION PROJECT
# ================================

# --------------------------------
# STEP 1 — Import Libraries
# --------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------
# STEP 2 — Load Dataset
# --------------------------------

df = pd.read_csv("house_prices_bangalore.csv")


# --------------------------------
# STEP 3 — Basic Data Inspection
# --------------------------------

print("First 5 Rows:")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# --------------------------------
# STEP 4 — Data Cleaning
# --------------------------------

# Remove missing values
df = df.dropna()

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Final dataset shape after cleaning
print("\nDataset Shape After Cleaning:")
print(df.shape)


# --------------------------------
# STEP 5 — Data Visualization
# --------------------------------

# Distribution of house prices
plt.figure(figsize=(8, 5))

df['price'].hist()

plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()


# --------------------------------
# STEP 6 — Correlation Heatmap
# --------------------------------

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap='coolwarm',
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()


# --------------------------------
# STEP 7 — Split Features & Target
# --------------------------------

# Features
X = df.drop('price', axis=1)

# Target variable
y = df['price']


# --------------------------------
# STEP 8 — Convert Categorical Data
# --------------------------------

X = pd.get_dummies(X, drop_first=True)

# Check data types
print("\nData Types:")
print(X.dtypes)


# --------------------------------
# STEP 9 — Train Test Split
# --------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# STEP 10 — Import ML Model
# --------------------------------

from sklearn.linear_model import LinearRegression


# --------------------------------
# STEP 11 — Create Model
# --------------------------------

model = LinearRegression()


# --------------------------------
# STEP 12 — Train Model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# STEP 13 — Model Accuracy
# --------------------------------

score = model.score(X_test, y_test)

print("\nModel Accuracy (R² Score):")
print(score)


# --------------------------------
# STEP 14 — Predictions
# --------------------------------

prediction = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(prediction[:5])


# --------------------------------
# STEP 15 — Error Calculation
# --------------------------------

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# MAE
mae = mean_absolute_error(y_test, prediction)

print("\nMean Absolute Error (MAE):")
print(mae)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, prediction))

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)


# --------------------------------
# STEP 16 — Actual vs Predicted Plot
# --------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, prediction)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()