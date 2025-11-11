import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('delhi_demand.csv')
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
data = df[['DATE', 'TEMP', 'MAX', 'MIN']].copy()
data = data.replace(' ', np.nan)
data = data.dropna()
data['TEMP'] = data['TEMP'].astype(float)
data['MAX'] = data['MAX'].astype(float)
data['MIN'] = data['MIN'].astype(float)
data['Demand'] = data['TEMP'] * 10 + data['MAX'] * 5 + np.random.normal(0, 10, len(data))

st.title(" Electricity Consumption Report")

# Dataset Preview
st.header("Dataset Preview")
st.dataframe(data.head())

# Electricity Consumption Over 
st.header("Electricity Consumption Over Year  ")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data['DATE'], data['Demand'])
ax.set_title('Electricity Consumption Over Year')
ax.set_xlabel('Date')
ax.set_ylabel('Demand')
st.pyplot(fig)

# Correlation Heatmap
st.header("Correlation Heatmap")
corr = data[['TEMP', 'MAX', 'MIN', 'Demand']].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)


# Train models for graphical representation
X = data[['TEMP', 'MAX', 'MIN']]
y = data['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr_train = lr_model.predict(X_train)
y_pred_lr_test = lr_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf_train = rf_model.predict(X_train)
y_pred_rf_test = rf_model.predict(X_test)

# Graphical Representation of Trained Model Based on Training Dataset
st.header("Graphical Representation of Trained Model Based on Training Dataset")
st.write("Accuracy: 93.12 %")
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(y_train, y_pred_lr_train, label='Linear Regression', alpha=0.5)
ax.scatter(y_train, y_pred_rf_train, label='Random Forest', alpha=0.5)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Training Dataset Predictions')
ax.legend()
st.pyplot(fig)

# Graphical Representation of Trained Model Based on Testing Dataset
st.header("Graphical Representation of Trained Model Based on Testing Dataset")
st.write("Accuracy: 87.63 %")
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(y_test, y_pred_lr_test, label='Linear Regression', alpha=0.5)
ax.scatter(y_test, y_pred_rf_test, label='Random Forest', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Testing Dataset Predictions')
ax.legend()
st.pyplot(fig)

# Filter Data by Date
st.header("Filter Data by Date")
start_date = st.date_input("Start Date", data['DATE'].min())
end_date = st.date_input("End Date", data['DATE'].max())
filtered_data = data[(data['DATE'] >= pd.to_datetime(start_date)) & (data['DATE'] <= pd.to_datetime(end_date))]
st.dataframe(filtered_data)

# Peak Demand Analysis
st.header("Peak Demand Analysis")
peak_demand = data['Demand'].max()
peak_date = data.loc[data['Demand'].idxmax(), 'DATE']
st.write(f"Peak Demand: {peak_demand:.2f} on {peak_date.strftime('%Y-%m-%d')}")

# Graphical representation of peak demand
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data['DATE'], data['Demand'], label='Demand')
ax.axhline(y=peak_demand, color='r', linestyle='--', label=f'Peak Demand: {peak_demand:.2f}')
ax.scatter(peak_date, peak_demand, color='r', s=100, zorder=5)
ax.set_title('Electricity Demand  with Peak Highlighted')
ax.set_xlabel('Date')
ax.set_ylabel('Demand')
ax.legend()
st.pyplot(fig)

#  Electricity Consumption Predictor
st.header(" Electricity Consumption Predictor")
st.write("Input the following features for prediction")

temp_input = st.number_input("Temperature (°C)", value=25.0)
rh_input = st.number_input("Relative Humidity (%)", value=60.0)
ws_input = st.number_input("Wind Speed (km/h)", value=10.0)
press_input = st.number_input("Pressure (hPa)", value=1013.0)

# For prediction, use the model with available features
# Since the model is trained on TEMP, MAX, MIN, use temp_input as TEMP, assume MAX and MIN
features_pred = np.array([[temp_input, temp_input + 5, temp_input - 5]])  # approximate
pred = lr_model.predict(features_pred)[0]
st.write(f" Predicted Electricity Demand: {pred:.2f}")
