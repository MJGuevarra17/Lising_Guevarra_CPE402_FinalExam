import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# URL for dataset - https://datahub.io/core/glacier-mass-balance
# Load dataset
df = pd.read_csv("glaciers.csv")

# Display basic info and data sample
print("=== Dataset Information ===")
print(df.info())
print("\n=== First 5 Rows of Data ===")
print(df.head())

# Quick statistical summary
print("\n=== Statistical Summary ===")
print(df.describe())

# Visualize the distribution of Mean cumulative mass balance
plt.figure(figsize=(8, 4))
sns.histplot(df['Mean cumulative mass balance'], bins=30, kde=True)
plt.title('Distribution of Mean Cumulative Mass Balance')
plt.xlabel('Mean Cumulative Mass Balance')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Data Cleaning
# Check for missing values
missing_values = df.isnull().sum()
print("\n=== Missing Values in Each Column ===")
print(missing_values)

# Remove rows with missing values if necessary
df.dropna(inplace=True)

# Check data types and convert if needed
print("\n=== Data Types ===")
print(df.dtypes)

# Visualization - Trends over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Mean cumulative mass balance', data=df, marker='o')
plt.title('Glacier Mass Balance Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Cumulative Mass Balance')
plt.grid()
plt.show()

# Prepare data for modeling
X = df[['Year']]  # Features
y = df['Mean cumulative mass balance']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Machine": SVR()
}

# Train models and store results
results = {}

print("\n=== Model Performance Results ===")
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {"MSE": mse, "R^2": r2}
    # Convert R² to percentage
    r2_percentage = r2 * 100
    print(f"{model_name} - MSE: {mse:.2f}, R^2: {r2_percentage:.2f}%")

# Model Evaluation
print("\n=== Model Evaluation Results ===")
for model_name, metrics in results.items():
    # Convert R² to percentage for evaluation
    r2_percentage = metrics["R^2"] * 100
    if metrics["R^2"] >= 0.85:
        print(f"{model_name} meets the performance requirement with R^2: {r2_percentage:.2f}%")
    else:
        print(f"{model_name} does not meet the performance requirement with R^2: {r2_percentage:.2f}%")
