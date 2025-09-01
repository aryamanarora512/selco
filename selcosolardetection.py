# Renewable Energy & Product Design Intern – Demo Project with Real Dataset
# Using NSRDB Bangalore Solar Irradiance Data
# created by Aryaman Arora
# audited by Sudarshan Singh Bisht -- sudarshans@selcofoundation.org

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch


# Replace this with your actual downloaded CSV file from NSRDB for Bangalore
df = pd.read_csv("nsrdb_india_2014.h5")

# Inspect data
print("Dataset preview:")
print(df.head())


# Ensure timestamp parsing
df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Features and target
features = ["DNI", "DHI", "Temperature", "Humidity", "Latitude", "Longitude"]
X = df[features]
y = df["GHI"]  # Global Horizontal Irradiance (target variable)


plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"][:200], y[:200], label="GHI (kWh/m²)", alpha=0.8)
plt.xlabel("Time")
plt.ylabel("GHI")
plt.title("Global Horizontal Irradiance (Sample: Bangalore)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.scatter(df["Temperature"], y, alpha=0.5)
plt.xlabel("Temperature (°C)")
plt.ylabel("GHI (kWh/m²)")
plt.title("Temperature vs Solar Radiation")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)

print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))



df["predicted_GHI"] = rf_model.predict(X)

# top sites by rank
best_sites = df.sort_values(by="predicted_GHI", ascending=False).head(5)
print("\nTop 5 Recommended Sites for Solar Panel Deployment:")
print(best_sites[["Latitude", "Longitude", "predicted_GHI", "Temperature", "Humidity"]])


# tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

print("\nPyTorch tensors ready:")
print("X_tensor shape:", X_tensor.shape)
print("y_tensor shape:", y_tensor.shape)
