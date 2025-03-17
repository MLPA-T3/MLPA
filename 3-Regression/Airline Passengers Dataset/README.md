### Continutation of airline passengers time series forecast using ARIMA (This time using Regression)

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the dataset
data = pd.read_csv(
    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
    index_col='Month', 
    parse_dates=True
)
data.index.freq = 'MS'  # monthly start frequency

# 2. Create a time index (0, 1, 2, ...)
data['time'] = np.arange(len(data))

# 3. Split into train (all but last 12) and test (last 12)
train = data.iloc[:-12]
test = data.iloc[-12:]

# 4. Prepare training data
X_train = train['time']
y_train = train['Passengers']

# Add a constant for the intercept term
X_train_const = sm.add_constant(X_train)

# 5. Fit the OLS model
ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())

# 6. Forecast on the test set
X_test_const = sm.add_constant(test['time'])
ols_forecast = ols_model.predict(X_test_const)

# 7. Evaluate forecast
mae = mean_absolute_error(test['Passengers'], ols_forecast)
rmse = np.sqrt(mean_squared_error(test['Passengers'], ols_forecast))
print("MAE:", mae)
print("RMSE:", rmse)

# 8. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Passengers'], label='Train')
plt.plot(test.index, test['Passengers'], label='Test', color='orange')
plt.plot(test.index, ols_forecast, label='OLS Forecast', color='red')
plt.title("Simple OLS Forecast")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()

```
![Capture](https://github.com/user-attachments/assets/1a5bb4dd-5255-498f-8522-5f058e0ab564)


![Figure 2025-03-04 134624](https://github.com/user-attachments/assets/8b43e56d-bdaa-487e-a3bd-be4b5c7e908e)

