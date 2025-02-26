# Comprehensive Time Series Forecasting with the Airline Passenger Dataset

## Abstract

Time series forecasting predicts future values based on past observations. In this document, we introduce the Airline Passenger dataset, perform exploratory analysis, and demonstrate a suite of classical forecasting methods. We cover univariate methods including:

- Autoregression (AR)
- Moving Average (MA)
- Autoregressive Moving Average (ARMA)
- Autoregressive Integrated Moving Average (ARIMA)
- Seasonal ARIMA (SARIMA)
- SARIMAX (Seasonal ARIMA with Exogenous Regressors)
- Simple Exponential Smoothing (SES)
- Holt Winter’s Exponential Smoothing (HWES)

Each section includes code that generates forecast plots alongside the historical data.

---

## 1. Introduction

The Airline Passenger dataset contains monthly totals of international airline passengers. It exhibits an upward trend and clear seasonal patterns—ideal for demonstrating various forecasting techniques. In the sections below, we load and visualize the data, then apply each forecasting method sequentially.

---

## 2. Data Loading and Exploratory Analysis

First, load the dataset and inspect its behavior.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(
    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
    index_col='Month', parse_dates=True
)
data.index.freq = 'MS'  # Monthly Start frequency

# Plot the observed data
plt.figure(figsize=(10, 6))
data['Passengers'].plot(title="Monthly Airline Passengers", label="Observed")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
```
![Figure 2025-02-26 124115](https://github.com/user-attachments/assets/7c6c085e-ea69-4df8-8137-782f31ca6af8)

*For some methods (like SARIMAX), we simulate an exogenous variable:*

```python
# Create an exogenous variable (e.g., a linear trend)
data['Trend'] = np.arange(len(data))
```

---

## 3. Forecasting Methods

Each section below explains the method and provides code that forecasts the next 12 months, along with a plot comparing the forecast to the historical data.

### 3.1 Autoregression (AR)

**Concept:**  
The AR model predicts the current value as a linear combination of its past observations.

**Code and Plot:**

```python
from statsmodels.tsa.ar_model import AutoReg

# Fit an AR model using 1 lag
ar_model = AutoReg(data['Passengers'], lags=1)
ar_fit = ar_model.fit()
# Forecast the next 12 months
ar_forecast = ar_fit.predict(start=len(data), end=len(data)+11)

# Create a forecast index
forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

# Plot the forecast
plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
pd.Series(ar_forecast.values, index=forecast_index).plot(label='AR Forecast', color='red')
plt.title("Autoregression (AR) Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/2b862b37-c3c6-42fe-9a68-a82261a0f545)

---

### 3.2 Moving Average (MA)

**Concept:**  
MA models forecast the next value based on past forecast errors.

**Code and Plot:**

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit an MA(1) model (set AR order to 0)
ma_model = ARIMA(data['Passengers'], order=(0, 0, 1))
ma_fit = ma_model.fit()
# Forecast the next 12 months
ma_forecast = ma_fit.predict(start=len(data), end=len(data)+11)

# Create a forecast index (reuse if needed)
forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

# Plot the forecast
plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
pd.Series(ma_forecast.values, index=forecast_index).plot(label='MA Forecast', color='red')
plt.title("Moving Average (MA) Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/52d8d0d7-1c6e-4003-98ed-e83b9bf12f18)

---

### 3.3 Autoregressive Moving Average (ARMA)

**Concept:**  
ARMA models combine AR and MA elements for stationary time series.

**Code and Plot:**

```python
# Fit an ARMA model by setting differencing d=0
arma_model = ARIMA(data['Passengers'], order=(2, 0, 1))
arma_fit = arma_model.fit()
# Forecast the next 12 months
arma_forecast = arma_fit.predict(start=len(data), end=len(data)+11)

# Create a forecast index (reuse if needed)
forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

# Plot the forecast
plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
pd.Series(arma_forecast.values, index=forecast_index).plot(label='ARMA Forecast', color='red')
plt.title("Autoregressive Moving Average (ARMA) Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/6e920272-6bd3-427d-a728-c21964182deb)

---

### 3.4 Autoregressive Integrated Moving Average (ARIMA)

**Concept:**  
ARIMA models incorporate differencing to address non-stationarity (trends).

**Code and Plot:**

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA model (order: p=1, d=1, q=1)
arima_model = ARIMA(data['Passengers'], order=(1, 1, 1))
arima_fit = arima_model.fit()
# Forecast the next 12 months
arima_forecast = arima_fit.forecast(steps=12)

plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
arima_forecast.plot(label='ARIMA Forecast', color='red')
plt.title("ARIMA Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/36ddcf69-c316-464b-9388-1648cb2d69b2)

---

### 3.5 Seasonal ARIMA (SARIMA)

**Concept:**  
SARIMA extends ARIMA by including seasonal components.

**Code and Plot:**

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit a SARIMA model with seasonal order (1,1,1,12)
sarima_model = SARIMAX(data['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
# Forecast the next 12 months
sarima_forecast = sarima_fit.get_forecast(steps=12).predicted_mean

plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
sarima_forecast.plot(label='SARIMA Forecast', color='red')
plt.title("SARIMA Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/1a2f3bf6-5a32-45e9-86d2-10cf9424574e)

---

### 3.6 SARIMAX (Seasonal ARIMA with Exogenous Regressors)

**Concept:**  
SARIMAX includes external predictors (exogenous variables) along with seasonal dynamics.

**Code and Plot:**

```python
# Use the synthetic 'Trend' variable as an exogenous regressor.
exog = data[['Trend']]
sarimax_model = SARIMAX(data['Passengers'], exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_fit = sarimax_model.fit(disp=False)
# Forecast 12 months ahead; create forecast exogenous variable (extend the trend)
exog_forecast = pd.DataFrame({'Trend': np.arange(len(data), len(data)+12)})
sarimax_forecast = sarimax_fit.get_forecast(steps=12, exog=exog_forecast).predicted_mean

plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
sarimax_forecast.plot(label='SARIMAX Forecast', color='red')
plt.title("SARIMAX Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/343bd227-a7b5-44c0-8c8e-c25ac7ddec22)

---

### 3.7 Simple Exponential Smoothing (SES)

**Concept:**  
SES applies exponentially decreasing weights to past observations, ideal for data with no trend or seasonality.

**Code and Plot:**

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ses_model = SimpleExpSmoothing(data['Passengers'])
ses_fit = ses_model.fit()
ses_forecast = ses_fit.forecast(12)

plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
ses_forecast.plot(label='SES Forecast', color='red')
plt.title("Simple Exponential Smoothing (SES) Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/e99f243f-ea1e-454e-aee0-2a08b2f947d2)

---

### 3.8 Holt Winter’s Exponential Smoothing (HWES)

**Concept:**  
HWES (or Triple Exponential Smoothing) captures level, trend, and seasonality simultaneously.

**Code and Plot:**

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

hwes_model = ExponentialSmoothing(data['Passengers'], trend='add', seasonal='mul', seasonal_periods=12)
hwes_fit = hwes_model.fit()
hwes_forecast = hwes_fit.forecast(12)

plt.figure(figsize=(10, 6))
data['Passengers'].plot(label='Observed')
hwes_forecast.plot(label='HWES Forecast', color='red')
plt.title("Holt Winter’s Exponential Smoothing (HWES) Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/8381b36c-4870-421a-b53d-fce4bf867b1b)

---

*End of document up to section 3.8.*
