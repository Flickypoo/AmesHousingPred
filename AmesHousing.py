import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

import plotly.express as px
import plotly.graph_objects as go

import joblib

import warnings
warnings.filterwarnings('ignore')

# Load the Dataset
try:
    df = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'train.csv' not found in the working directory.")
    exit()

print("\nFirst 5 rows of the dataset:")
print(df.head())

# Data Exploration and EDA

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Checking for Missing Values
print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Handling Missing Values

# Fill missing numerical features with median
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['SalePrice', 'Id']]  # Exclude target and ID

print("\nNumerical Columns (excluding 'SalePrice' and 'Id'):")
print(numerical_cols)

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        print(f"Filled missing values in '{col}' with median value {median}.")

# Fill missing categorical features with 'None'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna('None', inplace=True)
        print(f"Filled missing values in '{col}' with 'None'.")

# Verify no missing values remain
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().max())  # Should be 0

# Exploratory Data Analysis

plt.figure(figsize=(10,6))
sns.histplot(df['SalePrice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# Feature Engineering

le = LabelEncoder()

# Identify Binary and Multi-category Columns
binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
multi_cols = [col for col in categorical_cols if df[col].nunique() > 2]

print("\nBinary Categorical Columns:")
print(binary_cols)

print("\nMulti-category Categorical Columns:")
print(multi_cols)

#  Encode Binary Categorical Columns using LabelEncoder
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"Encoded binary column '{col}'.")

# Encode Multi-category Categorical Columns using One-Hot Encoding
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print("\nApplied One-Hot Encoding to multi-category columns.")

# Check for Remaining Categorical Columns
remaining_categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nRemaining Categorical Columns after Initial Encoding:", remaining_categorical_cols)

#  Handle Remaining Categorical Columns if Any
if remaining_categorical_cols:
    print("\nHandling Remaining Categorical Columns:")
    for col in remaining_categorical_cols:
        unique_values = df[col].nunique()
        print(f"Encoding column '{col}' with {unique_values} unique values.")
        
        if unique_values == 2:
            # Binary Encoding
            df[col] = le.fit_transform(df[col])
            print(f"Encoded binary column '{col}'.")
        else:
            # One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            print(f"Applied One-Hot Encoding to column '{col}'.")
    
    print("\nData after Handling Remaining Categorical Variables:")
    print(df.head())
else:
    print("\nNo remaining categorical columns to encode.")

#  Feature Scaling

# Define the features to scale
scaled_features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                  'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
                  '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea']

# Verify which scaled_features exist in the DataFrame
existing_scaled_features = [col for col in scaled_features if col in df.columns]

print("\nFeatures to be scaled:")
print(existing_scaled_features)

if existing_scaled_features:
    scaler = StandardScaler()
    df[existing_scaled_features] = scaler.fit_transform(df[existing_scaled_features])
    print("\nApplied StandardScaler to numerical features.")
else:
    print("\nNo features to scale. Check the 'scaled_features' list for correctness.")

#  Verify All Columns Are Numeric Before Correlation
print("\nData Types After Encoding and Scaling:")
print(df.dtypes.unique())

if not df.select_dtypes(include=['object']).empty:
    print("\nWarning: There are still non-numeric columns in the DataFrame.")
    print("Remaining Categorical Columns:", df.select_dtypes(include=['object']).columns.tolist())
else:
    print("\nAll columns are now numeric. Proceeding to compute the correlation matrix.")

# Compute the Correlation Matrix

if not df.select_dtypes(include=['object']).empty:
    print("\nError: Cannot compute correlation matrix with non-numeric columns. Please ensure all columns are numeric.")
    exit()
else:
    print("\nComputing Correlation Matrix...")
    corr = df.corr()
    
    #  Plotting the Correlation Heatmap
    plt.figure(figsize=(20,15))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Top 10 Features Correlated with SalePrice
    top_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(11)  # Includes 'SalePrice' itself
    print("\nTop 10 Features Correlated with SalePrice:")
    print(top_corr)
    
    plt.figure(figsize=(10,8))
    sns.barplot(x=top_corr.values[1:], y=top_corr.index[1:])  # Exclude 'SalePrice' from the plot
    plt.title('Top 10 Features Correlated with SalePrice')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.show()

# Preparing Data for Modeling

y = df['SalePrice']
X = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')  # Use errors='ignore' to prevent KeyError if 'Id' is absent

print("\nFeature Matrix (X) shape:", X.shape)
print("Target Vector (y) shape:", y.shape)

#  Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

#  Linear Regression
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"MAE: {mae_lr:.2f}")
print(f"MSE: {mse_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R2 Score: {r2_lr:.2f}")

# Decision Tree Regressor
print("\nTraining Decision Tree Regressor...")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Regressor Performance:")
print(f"MAE: {mae_dt:.2f}")
print(f"MSE: {mse_dt:.2f}")
print(f"RMSE: {rmse_dt:.2f}")
print(f"R2 Score: {r2_dt:.2f}")

#  Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor Performance:")
print(f"MAE: {mae_rf:.2f}")
print(f"MSE: {mse_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R2 Score: {r2_rf:.2f}")

#  Gradient Boosting Regressor
print("\nTraining Gradient Boosting Regressor...")
gb = GradientBoostingRegressor(random_state=42, n_estimators=100)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("Gradient Boosting Regressor Performance:")
print(f"MAE: {mae_gb:.2f}")
print(f"MSE: {mse_gb:.2f}")
print(f"RMSE: {rmse_gb:.2f}")
print(f"R2 Score: {r2_gb:.2f}")

# XGBoost Regressor
print("\nTraining XGBoost Regressor...")
xgbr = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
xgbr.fit(X_train, y_train)
y_pred_xgbr = xgbr.predict(X_test)
mae_xgbr = mean_absolute_error(y_test, y_pred_xgbr)
mse_xgbr = mean_squared_error(y_test, y_pred_xgbr)
rmse_xgbr = np.sqrt(mse_xgbr)
r2_xgbr = r2_score(y_test, y_pred_xgbr)

print("XGBoost Regressor Performance:")
print(f"MAE: {mae_xgbr:.2f}")
print(f"MSE: {mse_xgbr:.2f}")
print(f"RMSE: {rmse_xgbr:.2f}")
print(f"R2 Score: {r2_xgbr:.2f}")

#  Model Evaluation and Selection

model_performance = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'MAE': [mae_lr, mae_dt, mae_rf, mae_gb, mae_xgbr],
    'MSE': [mse_lr, mse_dt, mse_rf, mse_gb, mse_xgbr],
    'RMSE': [rmse_lr, rmse_dt, rmse_rf, rmse_gb, rmse_xgbr],
    'R2 Score': [r2_lr, r2_dt, r2_rf, r2_gb, r2_xgbr]
})

print("\nModel Performance Comparison:")
print(model_performance)

# Visualizing Model Performance
performance_melted = model_performance.melt(id_vars='Model', var_name='Metric', value_name='Value')

plt.figure(figsize=(12,8))
sns.barplot(x='Value', y='Model', hue='Metric', data=performance_melted)
plt.title('Model Performance Comparison')
plt.xlabel('Value')
plt.ylabel('Model')
plt.legend(title='Metric')
plt.show()

# Feature Importance Analysis

print("\nFeature Importance Analysis using XGBoost:")

importances = xgbr.feature_importances_
features = X.columns

# Create a DataFrame
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Top 20 Feature Importances from XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_xgbr, alpha=0.3, color='blue')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs. Predicted SalePrice')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Residual Analysis
residuals = y_test - y_pred_xgbr

plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True, bins=30, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# Save the XGBoost model
model_filename = 'xgboost_house_price_model.pkl'
joblib.dump(xgbr, model_filename)
print(f"\nModel saved successfully as '{model_filename}'.")

