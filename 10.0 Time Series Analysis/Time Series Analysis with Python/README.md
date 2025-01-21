**Time Series Analysis and Forecasting with Practical Python Examples**

![](https://miro.medium.com/v2/resize:fit:816/1*KXFPyusCXxfqNxhSi1MFwQ.png)

Time Series Data Analysis Image generated with DALL-E

Welcome to this comprehensive guide on time series data analytics and forecasting using Python. Whether you are a seasoned data analyst or a business analyst looking to dive deeper into time series analysis, this guide is tailored for you. We will walk you through the essentials of time series data, from understanding its fundamental components to applying sophisticated forecasting techniques.

![](https://miro.medium.com/v2/resize:fit:816/1*Om3uIUB6GrU93BLvjONLkQ.png)

Generative AI generated Image

**In this guide, you’ll learn:**

-   What time series data is and its unique characteristics.
-   How to preprocess and clean your data for accurate analysis.
-   Techniques for exploring and visualizing your data.
-   Methods for decomposing time series to understand underlying patterns.
-   Various forecasting methods, including classical approaches and modern machine learning techniques.
-   How to evaluate and select the best model for your data.
-   Practical implementation with real-world examples and Python code.

By the end of this guide, you’ll be equipped with the knowledge and tools to perform robust time series analysis and make accurate forecasts that can drive valuable insights for your business.

## Understanding Time Series Data

In this section, we’ll dive into what time series data is and why it’s essential in data analysis and forecasting.

![](https://miro.medium.com/v2/resize:fit:816/1*eFTiV0q4_AamT_gt5m8Nag.png)

Sample Timeseries Data Plots

**What is Time Series Data?**

Time series data is a sequence of data points collected or recorded at specific time intervals. Examples include daily stock prices, monthly sales figures, yearly climate data, and many more. The primary characteristic of time series data is its temporal order, meaning the sequence in which the data points are recorded matters.

**Unique Characteristics of Time Series Data**

Time series data has several unique characteristics that distinguish it from other types of data:

1.  **Trend**: This is the long-term movement or direction in the data. For example, the general increase in a company’s sales over several years.
2.  **Seasonality**: These are patterns that repeat at regular intervals, such as higher ice cream sales during the summer.
3.  **Cyclic Patterns**: Unlike seasonality, cyclic patterns are not of a fixed period. These could be influenced by economic cycles or other factors.
4.  **Irregular Components**: These are random or unpredictable variations in the data.

**Types of Time Series Data**

**Univariate vs. Multivariate Time Series**:

-   **Univariate**: A single variable or feature recorded over time (e.g., daily temperature).
-   **Multivariate**: Multiple variables recorded over time (e.g., daily temperature, humidity, and wind speed).

**Regular vs. Irregular Time Series**:

-   **Regular**: Data points are recorded at consistent time intervals (e.g., hourly, daily).
-   **Irregular**: Data points are recorded at inconsistent time intervals.

**Python Code Example**

Let’s create a simple time series dataset using Python to illustrate these concepts.

```python
import pandas as pd
import numpy as np

# Creating a sample time series data
date_range = pd.date_range(start='1/1/2020', periods=100, freq='D')
data = np.random.randn(100)
time_series = pd.Series(data, index=date_range)

print(time_series.head())
```

This code snippet creates a univariate time series data with 100 daily records starting from January 1, 2020.

> Understanding the basic characteristics and types of time series data is crucial because it helps in selecting the right analysis techniques and models. Recognizing patterns such as trends and seasonality can significantly improve forecasting accuracy.

## Preprocessing Time Series Data

Before diving into the analysis and forecasting, it’s essential to preprocess your time series data to ensure accuracy and reliability. This section will cover how to handle missing values, outliers, and transform your data for better analysis.

![](https://miro.medium.com/v2/resize:fit:816/1*NCRCcnbn0sxyQfQGt6aC2Q.png)

Timeseries Data Plots

**Data Collection and Cleaning**

Time series data often comes from various sources such as databases, APIs, or CSV files. The first step is to load your data into a suitable format, usually a Pandas DataFrame.

**Handling Missing Values and Outliers**

Missing values and outliers can significantly affect your analysis. Here are common methods to handle them:

1.  **Filling Missing Values**: Use methods like forward fill, backward fill, or interpolation.
2.  **Removing Outliers**: Detect and remove outliers using statistical methods like Z-score or IQR (Interquartile Range).

**Python Code Example**

```python
# Creating a time series with NaN values
time_series_with_nan = time_series.copy()
time_series_with_nan[::10] = np.nan  

# Filling missing values with forward fill
time_series_filled = time_series_with_nan.fillna(method='ffill')

# Importing zscore from scipy.stats
from scipy.stats import zscore

# Calculating z-scores and filtering outliers
z_scores = zscore(time_series_filled)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
time_series_no_outliers = time_series_filled[filtered_entries]
```

**Data Transformation**

Transforming your data can help in identifying patterns and making it ready for analysis. Common transformations include:

1.  **Smoothing**: Techniques like moving average can help in reducing noise.
2.  **Differencing**: Used to remove trends and seasonality.
3.  **Scaling and Normalization**: Ensure your data fits within a specific range for better model performance.

**Python Code Example**

```python
# Calculating a moving average with a window size of 5
moving_avg = time_series_no_outliers.rolling(window=5).mean()

# Differencing the series to remove trends
differenced_series = time_series_no_outliers.diff().dropna()

# Importing StandardScaler from sklearn for scaling
from sklearn.preprocessing import StandardScaler

# Scaling the series for better model performance
scaler = StandardScaler()
scaled_series = scaler.fit_transform(time_series_no_outliers.values.reshape(-1, 1))
```

> Preprocessing is a crucial step that ensures your data is clean and well-prepared for analysis. Proper handling of missing values and outliers, along with necessary transformations, can significantly enhance the accuracy of your forecasts.

## Exploratory Data Analysis (EDA)

In this section, we’ll explore various techniques to understand and visualize your time series data. EDA helps uncover patterns, trends, and relationships that can inform your forecasting models.

![](https://miro.medium.com/v2/resize:fit:816/1*lISmP_12HZzbIL2_PL-66g.png)

EDA With Timeseries Dataset

**Visualization Techniques**

Visualizing your time series data is crucial to grasp its underlying patterns and characteristics. Here are some essential visualization techniques:

1.  **Line Plots**: The most straightforward way to visualize time series data.
2.  **Seasonal Plots**: To identify seasonal patterns.
3.  **Autocorrelation Plots**: To check the correlation of the series with its past values.

**Python Code Example**

Let’s use Python to create these visualizations.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set seed for reproducibility
np.random.seed(0)

# Generate a date range
date_range = pd.date_range(start='1/1/2010', periods=120, freq='M')

# Create components for the time series
trend = np.linspace(50, 150, 120)
seasonality = 10 + 20 * np.sin(np.linspace(0, 3.14 * 2, 120))
noise = np.random.normal(scale=10, size=120)

# Combine components to create the time series
data = trend + seasonality + noise
time_series = pd.Series(data, index=date_range)

# Introduce missing values and fill them
time_series[::15] = np.nan
time_series = time_series.fillna(method='ffill')

# Calculate moving average, differencing, and monthly averages
moving_avg = time_series.rolling(window=12).mean()
differenced_series = time_series.diff().dropna()
time_series_monthly = time_series.resample('M').mean()

# Plot the time series data
plt.figure(figsize=(15, 18))

# Plot original time series
plt.subplot(4, 1, 1)
plt.plot(time_series, label='Original Time Series', color='blue')
plt.title('Time Series Line Plot')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)

# Plot seasonal pattern
plt.subplot(4, 1, 2)
plt.plot(time_series_monthly, label='Monthly Averaged Time Series', color='green')
plt.title('Seasonal Plot')
plt.xlabel('Date')
plt.ylabel('Monthly Average Sales')
plt.legend()
plt.grid(True)

# Plot autocorrelation
plt.subplot(4, 1, 3)
cleaned_time_series = time_series.dropna()
plot_acf(cleaned_time_series, lags=20, ax=plt.gca())
plt.title('Autocorrelation Plot')
plt.grid(True)

# Plot partial autocorrelation
plt.subplot(4, 1, 4)
plot_pacf(cleaned_time_series, lags=20, ax=plt.gca())
plt.title('Partial Autocorrelation Plot')
plt.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('./eda_visuals.png')
plt.show()
```

EDA provides insights into the structure and characteristics of your time series data. For example:

-   **Line Plots**: Help identify overall trends and any apparent seasonality.
-   **Seasonal Plots**: Highlight repeating patterns at regular intervals.
-   **Autocorrelation and Partial Autocorrelation Plots**: Show the relationship between current and past values of the series, indicating potential lags to include in models.

## Time Series Decomposition

Time series decomposition is a crucial step in understanding the underlying components of your data. By breaking down a time series into its constituent parts, you can gain insights into the trend, seasonality, and residual (irregular) components. This process helps in better understanding and forecasting the data.

![](https://miro.medium.com/v2/resize:fit:816/1*GKeDg0KYoSnZ5sAmggsUhQ.png)

**Decomposition Techniques**

**Decomposition Techniques**

There are two main models for decomposing a time series: additive and multiplicative.

**1\. Additive Model**: This model assumes that the components add together to produce the time series:

```
    Y(t) = T(t) + S(t) + R(t)
```

where Y(t) is the observed time series, T(t) is the trend component, S(t) is the seasonal component, and R(t) is the residual component.

**2\. Multiplicative Model**: This model assumes that the components multiply together to produce the time series:

```
    Y(t) = T(t) × S(t) × R(t)
```

The multiplicative model is useful when the seasonal variations are proportional to the level of the trend.

**Using Python to Decompose Time Series**

Let’s decompose the time series using Python’s `statsmodels` library. We'll use the realistic dummy dataset we created earlier.

**Python Code Example**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Setting the random seed
np.random.seed(0)

# Creating time series data
date_range = pd.date_range(start='1/1/2010', periods=120, freq='M')
trend = np.linspace(50, 150, 120)  
seasonality = 10 + 20 * np.sin(np.linspace(0, 3.14 * 2, 120))  
noise = np.random.normal(scale=10, size=120)  
data = trend + seasonality + noise
time_series = pd.Series(data, index=date_range)

# Handling missing values
time_series[::15] = np.nan
time_series = time_series.fillna(method='ffill')
time_series = time_series.dropna()

# Decomposing the time series
decomposition = seasonal_decompose(time_series, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plotting the decomposition
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(time_series, label='Original Time Series', color='blue')
plt.title('Original Time Series')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend Component', color='orange')
plt.title('Trend Component')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal Component', color='green')
plt.title('Seasonal Component')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(residual, label='Residual Component', color='red')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.savefig('./visuals/time_series_decomposition.png')
plt.show()

```

Decomposing the time series helps in identifying:

-   **Trend**: The long-term progression of the series.
-   **Seasonality**: The repeating short-term cycle in the series.
-   **Residuals**: The remaining noise after removing the trend and seasonality.

These components can be analyzed separately to better understand the behavior of the time series and to improve forecasting models.

## Time Series Forecasting Methods

Forecasting is the process of making predictions about future values based on historical time series data. There are various methods to achieve this, ranging from simple statistical models to advanced machine learning techniques. In this section, we’ll explore some of the most commonly used forecasting methods.

![](https://miro.medium.com/v2/resize:fit:816/1*zBjs0oKCWzFD37s_3zvD4A.png)

Time Series Forecasting

**Classical Methods**

1.  Moving Average (MA)
2.  Autoregressive (AR) Models
3.  Autoregressive Integrated Moving Average (ARIMA)

**Advanced Methods**

1.  Seasonal ARIMA (SARIMA)
2.  Exponential Smoothing State Space Model (ETS)

**Machine Learning Approaches**

1.  Linear Regression
2.  Decision Trees and Random Forests
3.  Support Vector Machines
4.  Neural Networks (RNN, LSTM)

Let’s focus on ARIMA and SARIMA models for this section, as they are widely used and relatively straightforward to implement with Python.

**Using Python for Time Series Forecasting**

We’ll use the `statsmodels` library to demonstrate how to fit ARIMA and SARIMA models to our time series data.

**Python Code Example**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Setting the random seed
np.random.seed(0)

# Creating time series data
date_range = pd.date_range(start='1/1/2010', periods=120, freq='M')
trend = np.linspace(50, 150, 120)
seasonality = 10 + 20 * np.sin(np.linspace(0, 3.14 * 2, 120))
noise = np.random.normal(scale=10, size=120)
data = trend + seasonality + noise
time_series = pd.Series(data, index=date_range)

# Handling missing values
time_series[::15] = np.nan
time_series = time_series.fillna(method='ffill')
time_series = time_series.dropna()

# Fitting ARIMA model
arima_model = ARIMA(time_series, order=(1, 1, 1))
arima_fit = arima_model.fit()
print(arima_fit.summary())

# Fitting SARIMA model
sarima_model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# Forecasting with ARIMA
arima_forecast = arima_fit.get_forecast(steps=12)
arima_forecast_index = pd.date_range(start=time_series.index[-1], periods=12, freq='M')
arima_forecast_series = pd.Series(arima_forecast.predicted_mean, index=arima_forecast_index)

# Forecasting with SARIMA
sarima_forecast = sarima_fit.get_forecast(steps=12)
sarima_forecast_index = pd.date_range(start=time_series.index[-1], periods=12, freq='M')
sarima_forecast_series = pd.Series(sarima_forecast.predicted_mean, index=sarima_forecast_index)

# Plotting the forecasts
plt.figure(figsize=(15, 8))
plt.plot(time_series, label='Original Time Series', color='blue')
plt.plot(arima_forecast_series, label='ARIMA Forecast', color='orange')
plt.plot(sarima_forecast_series, label='SARIMA Forecast', color='green')
plt.title('Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./visuals/time_series_forecasting.png')
plt.show()

```

Please note that:

1.  **ARIMA Model**: Suitable for non-seasonal data, ARIMA can model the autocorrelations in the data.
2.  **SARIMA Model**: Extends ARIMA to handle seasonality, making it ideal for data with seasonal patterns.

## Model Evaluation and Selection

Selecting the best forecasting model is crucial for making accurate predictions. In this section, we’ll explore various techniques and metrics for evaluating time series models and selecting the most suitable one.

**Evaluation Metrics**

**1\. Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction.

![](https://miro.medium.com/v2/resize:fit:450/1*18P-UcpFhyzNhyyMOcjxqQ.png)

2\. **Mean Squared Error (MSE)**: Measures the average of the squares of the errors, giving more weight to larger errors.

![](https://miro.medium.com/v2/resize:fit:364/1*-lwGgIHLQ-ANU857VOUAPQ.png)

**3\. Root Mean Squared Error (RMSE)**: The square root of the mean squared error.

![](https://miro.medium.com/v2/resize:fit:310/1*NWaih5OPg8G3To2YLARkTA.png)

4\. **Mean Absolute Percentage Error (MAPE)**: Measures the average absolute percentage error.

![](https://miro.medium.com/v2/resize:fit:387/1*o55fh_PlaIOlvJ6LookJbA.png)

**Cross-Validation Techniques**

1.  **Rolling Forecast Origin**: A method where the model is re-estimated for each prediction point, and the forecast is made for the next time step.
2.  **Time Series Split**: Splitting the time series data into training and testing sets based on time.

**Model Diagnostics**

1.  **Residual Analysis**: Check if the residuals (errors) from the model are white noise (i.e., they should be uncorrelated and normally distributed with a mean of zero).
2.  **Autocorrelation of Residuals**: Use autocorrelation plots to check for patterns in the residuals.

Let’s evaluate the ARIMA and SARIMA models we fitted in the previous section.

**Python Code Example**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setting the random seed
np.random.seed(0)

# Creating time series data
date_range = pd.date_range(start='1/1/2010', periods=120, freq='M')
trend = np.linspace(50, 150, 120)
seasonality = 10 + 20 * np.sin(np.linspace(0, 3.14 * 2, 120))
noise = np.random.normal(scale=10, size=120)
data = trend + seasonality + noise
time_series = pd.Series(data, index=date_range)

# Handling missing values
time_series[::15] = np.nan
time_series = time_series.fillna(method='ffill')
time_series = time_series.dropna()

# Fitting ARIMA model
arima_model = ARIMA(time_series, order=(1, 1, 1))
arima_fit = arima_model.fit()

# Fitting SARIMA model
sarima_model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Forecasting with ARIMA
arima_forecast = arima_fit.get_forecast(steps=12)
arima_forecast_series = arima_forecast.predicted_mean
arima_forecast_ci = arima_forecast.conf_int()

# Forecasting with SARIMA
sarima_forecast = sarima_fit.get_forecast(steps=12)
sarima_forecast_series = sarima_forecast.predicted_mean
sarima_forecast_ci = sarima_forecast.conf_int()

# Actual values
actual = time_series[-12:]

# Evaluation metrics for ARIMA
arima_mae = mean_absolute_error(actual, arima_forecast_series[:12])
arima_mse = mean_squared_error(actual, arima_forecast_series[:12])
arima_rmse = np.sqrt(arima_mse)
arima_mape = np.mean(np.abs((actual - arima_forecast_series[:12]) / actual)) * 100

# Evaluation metrics for SARIMA
sarima_mae = mean_absolute_error(actual, sarima_forecast_series[:12])
sarima_mse = mean_squared_error(actual, sarima_forecast_series[:12])
sarima_rmse = np.sqrt(sarima_mse)
sarima_mape = np.mean(np.abs((actual - sarima_forecast_series[:12]) / actual)) * 100

# Print metrics
print(f"ARIMA MAE: {arima_mae}, MSE: {arima_mse}, RMSE: {arima_rmse}, MAPE: {arima_mape}")
print(f"SARIMA MAE: {sarima_mae}, MSE: {sarima_mse}, RMSE: {sarima_rmse}, MAPE: {sarima_mape}")

# Plotting residuals
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(arima_fit.resid, label='ARIMA Residuals', color='blue')
plt.title('ARIMA Model Residuals')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sarima_fit.resid, label='SARIMA Residuals', color='green')
plt.title('SARIMA Model Residuals')
plt.legend()

plt.tight_layout()
plt.savefig('./visuals/model_evaluation_residuals.png')
plt.show()

```

```
ARIMA Metrics:

MAE: 6.740723561298947
MSE: 86.05719515259203
RMSE: 9.276701738904405
MAPE: NaN (likely due to division by zero or very small actual values)
SARIMA Metrics:

MAE: 11.684219153028662
MSE: 195.5162161410725
RMSE: 13.982711330105921
MAPE: NaN (same as above, potentially from invalid division)
```

![](https://miro.medium.com/v2/resize:fit:816/1*sZ3dvFKZ9T072BoXs8nrkA.png)

Residual plots

**Note That:**

The visual above shows the residuals from the ARIMA and SARIMA models. The evaluation metrics are as follows:

**ARIMA**:

-   MAE: 6.74
-   MSE: 86.06
-   RMSE: 9.28
-   MAPE: NaN (due to division by zero or very small actual values)

**SARIMA**:

-   MAE: 11.68
-   MSE: 195.52
-   RMSE: 13.98
-   MAPE: NaN (due to division by zero or very small actual values)

The residual plots indicate how well the models fit the data, with the goal being that the residuals should ideally resemble white noise.

1.  **Evaluation Metrics**: Provide a quantitative measure of the model’s performance.
2.  **Residual Analysis**: Helps in diagnosing issues with the model and ensuring that the residuals are uncorrelated and normally distributed.

## Practical Implementation

In this section, we’ll walk through a practical implementation of time series forecasting using a real-world example. We’ll demonstrate the entire workflow, from data loading and preprocessing to model building, evaluation, and forecasting.

**Tools and Libraries**

We’ll use the following Python libraries:

-   **Pandas** for data manipulation
-   **NumPy** for numerical operations
-   **Matplotlib** for data visualization
-   **statsmodels** for statistical modeling
-   **scikit-learn** for machine learning algorithms

**Step-by-Step Case Study**

Let’s assume we have a dataset of monthly sales data for a retail store over the last 10 years. We’ll use this data to forecast future sales.

1.  **Load the Data**
2.  **Preprocess the Data**
3.  **Exploratory Data Analysis (EDA)**
4.  **Time Series Decomposition**
5.  **Model Building and Forecasting**
6.  **Model Evaluation**
7.  **Plotting the Forecast**

**Python Code Example**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load the Data
np.random.seed(0)
date_range = pd.date_range(start='1/1/2010', periods=120, freq='M')
trend = np.linspace(50, 150, 120)  # Linear trend
seasonality = 10 + 20 * np.sin(np.linspace(0, 3.14 * 2, 120))  # Seasonal component
noise = np.random.normal(scale=10, size=120)  # Random noise
data = trend + seasonality + noise
time_series = pd.Series(data, index=date_range)

# Step 2: Preprocess the Data
time_series[::15] = np.nan
time_series = time_series.fillna(method='ffill')
time_series = time_series.dropna()

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Monthly Sales')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 4: Time Series Decomposition
decomposition = seasonal_decompose(time_series, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(15, 12))
plt.subplot(4, 1, 1)
plt.plot(time_series, label='Original Time Series', color='blue')
plt.title('Original Time Series')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend Component', color='orange')
plt.title('Trend Component')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal Component', color='green')
plt.title('Seasonal Component')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(residual, label='Residual Component', color='red')
plt.title('Residual Component')
plt.legend()
plt.tight_layout()
plt.show()

# Step 5: Model Building and Forecasting
arima_model = ARIMA(time_series, order=(1, 1, 1))
arima_fit = arima_model.fit()

sarima_model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

arima_forecast = arima_fit.get_forecast(steps=12)
arima_forecast_series = arima_forecast.predicted_mean
arima_forecast_ci = arima_forecast.conf_int()

sarima_forecast = sarima_fit.get_forecast(steps=12)
sarima_forecast_series = sarima_forecast.predicted_mean
sarima_forecast_ci = sarima_forecast.conf_int()

# Step 6: Model Evaluation
actual = time_series[-12:]

arima_mae = mean_absolute_error(actual, arima_forecast_series)
arima_mse = mean_squared_error(actual, arima_forecast_series)
arima_rmse = np.sqrt(arima_mse)
arima_mape = np.mean(np.abs((actual - arima_forecast_series) / actual)) * 100

sarima_mae = mean_absolute_error(actual, sarima_forecast_series)
sarima_mse = mean_squared_error(actual, sarima_forecast_series)
sarima_rmse = np.sqrt(sarima_mse)
sarima_mape = np.mean(np.abs((actual - sarima_forecast_series) / actual)) * 100

print(f"ARIMA MAE: {arima_mae}, MSE: {arima_mse}, RMSE: {arima_rmse}, MAPE: {arima_mape}")
print(f"SARIMA MAE: {sarima_mae}, MSE: {sarima_mse}, RMSE: {sarima_rmse}, MAPE: {sarima_mape}")

# Step 7: Plotting the Forecast
plt.figure(figsize=(15, 8))
plt.plot(time_series, label='Original Time Series', color='blue')
plt.plot(arima_forecast_series.index, arima_forecast_series, label='ARIMA Forecast', color='orange')
plt.plot(sarima_forecast_series.index, sarima_forecast_series, label='SARIMA Forecast', color='green')
plt.fill_between(arima_forecast_series.index, arima_forecast_ci.iloc[:, 0], arima_forecast_ci.iloc[:, 1], color='orange', alpha=0.3)
plt.fill_between(sarima_forecast_series.index, sarima_forecast_ci.iloc[:, 0], sarima_forecast_ci.iloc[:, 1], color='green', alpha=0.3)
plt.title('Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./visuals/practical_implementation_forecasting.png')
plt.show()

```

![](https://miro.medium.com/v2/resize:fit:816/1*-fAronib7AkQOl-uu6jzrg.png)

![](https://miro.medium.com/v2/resize:fit:816/1*GKeDg0KYoSnZ5sAmggsUhQ.png)

This code will walk you through the entire process of time series forecasting using a realistic dummy dataset, including data loading, preprocessing, EDA, decomposition, model building, evaluation, and plotting the forecast.

-   **End-to-End Workflow**: Demonstrates the complete process of time series forecasting.
-   **Practical Application**: Shows how to apply the techniques learned in previous sections to a real-world scenario.

## Best Practices and Common Pitfalls

In this final section, we’ll discuss some best practices to follow when performing time series analysis and forecasting, as well as common pitfalls to avoid.

**Best Practices**

1.  **Understand Your Data**: Before diving into analysis, take the time to understand the nature of your time series data. Recognize patterns such as trends, seasonality, and cycles.
2.  **Proper Preprocessing**: Always clean your data to handle missing values and outliers. Ensure your data is stationary if required by the model you choose.
3.  **Model Selection**: Choose the right model based on the characteristics of your data. Use simpler models like ARIMA for non-seasonal data and SARIMA for seasonal data.
4.  **Cross-Validation**: Use cross-validation techniques to evaluate the performance of your models. Rolling forecast origin and time series split are common methods.
5.  **Residual Analysis**: After fitting your model, analyze the residuals to ensure they are white noise. This indicates that the model has captured all the underlying patterns in the data.
6.  **Regular Updates**: Time series data can change over time, so it’s crucial to regularly update your models with new data to maintain their accuracy.
7.  **Automate Where Possible**: Use automated tools and scripts to streamline your workflow, especially for data preprocessing and model evaluation.

**Common Pitfalls**

1.  **Overfitting**: Creating a model that is too complex can lead to overfitting, where the model performs well on the training data but poorly on new data. Use simpler models and regularization techniques to avoid this.
2.  **Ignoring Seasonality**: Failing to account for seasonal patterns can lead to inaccurate forecasts. Always check for seasonality in your data and use appropriate models like SARIMA if necessary.
3.  **Neglecting Model Diagnostics**: Skipping residual analysis and other diagnostics can lead to misleading results. Always perform thorough diagnostics to validate your model.
4.  **Using Inappropriate Metrics**: Different metrics provide different insights. Using only one metric can give an incomplete picture of model performance. Use multiple evaluation metrics like MAE, MSE, RMSE, and MAPE.
5.  **Data Leakage**: Ensure that future data does not influence the model training. This can lead to over-optimistic performance estimates. Properly split your data into training and test sets.
6.  **Static Models for Dynamic Data**: Static models can become obsolete as new data becomes available. Regularly update your models to adapt to new trends and patterns.

