# ==========================================
# OASIS INFOBYTE INTERNSHIP
# TASK 3 - CAR PRICE PREDICTION WITH ML
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("predictions", exist_ok=True)
os.makedirs("model", exist_ok=True)

print("Folders Created Successfully!")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("D:\Oasis Internship 2026\Car Price Prediction\car data.csv")

print("\n" + "="*60)
print("DATASET LOADED SUCCESSFULLY")
print("="*60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# DATA VISUALIZATION
# ==========================================

print("\nGenerating Visualizations...")

# 1. Selling Price Distribution

plt.figure(figsize=(8,5))
sns.histplot(df['Selling_Price'], bins=20, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig(
    "outputs/selling_price_distribution.png",
    dpi=300
)
plt.show()

# 2. Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=300
)
plt.show()

# ==========================================
# DATA PREPROCESSING
# ==========================================

print("\nPerforming Data Preprocessing...")

le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

# Remove Car Name Column

df.drop('Car_Name', axis=1, inplace=True)

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# MODEL TRAINING
# ==========================================

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Mean Absolute Error (MAE) : {mae:.2f}")
print(f"Mean Squared Error (MSE)  : {mse:.2f}")
print(f"Root Mean Squared Error   : {rmse:.2f}")
print(f"R² Score                  : {r2:.4f}")

# ==========================================
# SAVE PREDICTIONS
# ==========================================

comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

comparison.to_csv(
    "predictions/predicted_prices.csv",
    index=False
)

print("\nPredictions Saved Successfully!")

print("\nSample Predictions:")
print(comparison.head(10))

# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature Importance:")
print(importance)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "model/car_price_prediction_model.pkl"
)

print("\nModel Saved Successfully!")

# ==========================================
# PROJECT COMPLETED
# ==========================================

print("\n" + "="*60)
print("CAR PRICE PREDICTION PROJECT COMPLETED")
print("="*60)

print("\nFiles Generated:")

print("""
outputs/
├── selling_price_distribution.png
├── correlation_heatmap.png
├── actual_vs_predicted.png
└── feature_importance.png

predictions/
└── predicted_prices.csv

model/
└── car_price_prediction_model.pkl
""")

print("All outputs saved successfully!")