# 🚗 Car Price Prediction with Machine Learning

## 📌 Project Overview

This project focuses on predicting the selling price of cars using Machine Learning techniques. Various factors such as present price, fuel type, transmission type, kilometers driven, and ownership details are analyzed to estimate the selling price of a vehicle.

The model is trained using Random Forest Regression and evaluated using standard performance metrics.

This project was completed as part of the **Oasis Infobyte Data Science Internship (Task 3)**.

---

## 🎯 Objectives

- Analyze car dataset
- Perform data preprocessing
- Visualize important patterns and relationships
- Train a Machine Learning model
- Predict car prices accurately
- Evaluate model performance
- Save predictions and trained model

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📂 Project Structure

```text
OIBSIP_Task3_CarPricePrediction/
│
├── car data.csv
├── car_price_prediction.py
├── README.md
│
├── outputs/
│   ├── selling_price_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── predictions/
│   └── predicted_prices.csv
│
└── model/
    └── car_price_prediction_model.pkl
```

---

## 📊 Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values
- Encoded categorical variables
- Removed unnecessary columns
- Split data into training and testing sets

---

## 🤖 Machine Learning Model

### Random Forest Regressor

The Random Forest algorithm was used because:

- Handles non-linear relationships effectively
- Reduces overfitting
- Provides feature importance analysis
- Produces high prediction accuracy

---

## 📈 Evaluation Metrics

The model performance was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📷 Visualizations Generated

### Selling Price Distribution
Shows how car selling prices are distributed across the dataset.

### Correlation Heatmap
Displays relationships among numerical features.

### Actual vs Predicted Prices
Compares actual selling prices with model predictions.

### Feature Importance
Highlights the most influential features affecting car prices.

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/OIBSIP_Task3_CarPricePrediction.git
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Run Project

```bash
python car_price_prediction.py
```

---

## 📌 Results

The model successfully predicts car prices based on multiple vehicle characteristics and achieves strong predictive performance using Random Forest Regression.

---

## 🎓 Internship Details

**Internship Provider:** Oasis Infobyte

**Domain:** Data Science

**Task:** Task 3 – Car Price Prediction with Machine Learning

---

## 👩‍💻 Author

Pallavi P

Data Science & Machine Learning Enthusiast
