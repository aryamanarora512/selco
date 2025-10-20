Renewable Energy & Product Design Intern – Demo Project with Real Dataset
Using NSRDB Bangalore Solar Irradiance Data
Created by Aryaman Arora
Audited by Sudarshan Singh Bisht
Overview
This project demonstrates a solar radiation prediction model developed as part of Aryaman Arora’s internship with the Technology Incubation Team at Selco Foundation (Bangalore, India).
The model leverages Random Forest regression and PyTorch tensors to analyze solar irradiance data from the National Solar Radiation Database (NSRDB) and identify optimal deployment sites for solar panels in the Bangalore region.
The insights from this model supported Selco’s renewable energy initiatives and guided the deployment of a solar-powered milking machine, improving energy access and livelihoods for underprivileged farmers.
Objectives
Perform exploratory analysis of solar irradiance and environmental data from NSRDB.
Predict Global Horizontal Irradiance (GHI) using Random Forest decision trees.
Identify top-performing geographic sites for solar panel deployment.
Visualize correlations between temperature, humidity, and solar output.
Prepare PyTorch-ready tensors for future deep learning model development.
Dataset
Source: National Solar Radiation Database (NSRDB), India (2014 dataset).
Location: Bangalore region.
Key Variables:
DNI – Direct Normal Irradiance
DHI – Diffuse Horizontal Irradiance
GHI – Global Horizontal Irradiance (Target)
Temperature, Humidity, Latitude, Longitude
Methodology
1. Data Loading and Preprocessing
The dataset is read from a local NSRDB .h5 file and time-indexed using year, month, day, hour, and minute columns.
df = pd.read_csv("nsrdb_india_2014.h5")
df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
2. Visualization
Time series and scatter plots are generated using Matplotlib to analyze trends and relationships in solar irradiance.
plt.plot(df["timestamp"][:200], df["GHI"][:200])
plt.scatter(df["Temperature"], df["GHI"])
3. Model Training
A RandomForestRegressor is trained to predict GHI based on weather and geographic variables.
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
4. Evaluation
Performance is assessed using Mean Squared Error (MSE) and R² metrics.
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
5. Site Ranking
Predicted GHI values are used to rank and identify top-performing solar deployment sites.
best_sites = df.sort_values(by="predicted_GHI", ascending=False).head(5)
6. PyTorch Integration
Data is converted to tensors for downstream deep learning workflows.
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
Results
Achieved strong predictive accuracy for solar irradiance in the Bangalore region.
Visualized temperature and humidity correlations with solar radiation.
Identified top 5 optimal deployment sites based on predicted GHI.
Generated PyTorch tensors for model scalability and integration with neural networks.
Impact
This project directly informed Selco Foundation’s renewable energy initiatives by providing a data-driven framework for solar infrastructure planning.
The findings supported the deployment of a solar-powered milking machine, improving productivity and income stability for five underprivileged farmers in rural Karnataka.
Tools and Technologies
Python, Pandas, Matplotlib, PyTorch
Scikit-learn (Random Forest Regressor)
NSRDB Solar Data (Bangalore region, 2014)
Author
Aryaman Arora
Renewable Energy & Product Design Intern – Selco Foundation, Technology Incubation Team
Duration: February 2023 – August 2023
Location: Bangalore, India
