import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the data
df = pd.read_csv('house_prices_practice.csv')

# Basic info
print("=== House Price Dataset Loaded ===")
print(f"Total Records: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
# Select features and target
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'FullBath', 'BedroomAbvGr', 'LotArea']
target = 'SalePrice'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n=== Data Split ===")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"R2 Score: {r2:.2f}")
# Predict house price
print("\n=== House Price Predictor ===")
print("Enter house details to get a price prediction!\n")

overall_qual = int(input("Overall Quality (1-10): "))
gr_liv_area = int(input("Living Area (sq ft): "))
garage_cars = int(input("Garage Capacity (0-4): "))
total_bsmt = int(input("Basement Area (sq ft): "))
year_built = int(input("Year Built: "))
full_bath = int(input("Number of Bathrooms: "))
bedrooms = int(input("Number of Bedrooms: "))
lot_area = int(input("Lot Area (sq ft): "))

# Make prediction
input_data = pd.DataFrame([[overall_qual, gr_liv_area, garage_cars, total_bsmt, year_built, full_bath, bedrooms, lot_area]], columns=features)
predicted_price = model.predict(input_data)[0]

print(f"\n🏠 Predicted House Price: ${predicted_price:,.0f}")