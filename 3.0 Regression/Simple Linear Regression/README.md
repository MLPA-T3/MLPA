# **Linear Regression Implementation in Python**

Regression is a supervised learning technique used to solve regression problems. It predicts continuous values such as temperature, price, sales, salary, age, etc. Linear regression is commonly used to find a linear relationship between a target and one or more predictors. This lesson introduces its concepts and implementation.

---

## **What is Linear Regression?**
Linear regression predicts the target variable by fitting the best linear relationship between the dependent (target) and independent (predictor) variables. It is used for:
- Forecasting
- Identifying cause-and-effect relationships between variables.

### **Assumptions of Linear Regression**
1. **Linear Relationship**:
   - The independent variables should be linearly related to the dependent variables.
   - Checked via scatter plots, pair plots, or heatmaps.

2. **Normal Distribution**:
   - The data should follow a normal distribution.
   - Verified using Q-Q plots or histograms.

3. **No Multicollinearity**:
   - Independent variables should not be highly correlated with each other.
   - Detected using correlation matrices or Variance Inflation Factor (VIF).

4. **Mean of Residuals Should Be Zero**:
   - Residual = Actual value - Predicted value.

5. **Residuals Should Be Normally Distributed**:
   - Verified using Q-Q plots.

6. **Homoscedasticity**:
   - The variance of residuals should be constant across all levels of the independent variable.
   - Checked with residual vs. fitted plots.

7. **No Auto-correlation**:
   - Residuals should be independent of each other.
   - Detected using the Durbin-Watson test or ACF plots.

---

## **Types of Linear Regression**

### **1. Simple Linear Regression**
- Involves one independent variable (X) and one dependent variable (Y).
- **Equation**: 
  
  \[ Y = \beta_0 + \beta_1X + e \]

  Where:
  - \( Y \): Dependent variable (target)
  - \( X \): Independent variable (predictor)
  - \( \beta_0 \): Intercept
  - \( \beta_1 \): Slope of the line
  - \( e \): Error term

- **Example**: Predicting sales based on money spent on TV advertisements.

---

## **Simple Linear Regression Implementation in Python**

### **Problem Statement**
Build a Simple Linear Regression model to predict sales based on money spent on TV advertising.

### **1. Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation.
- **Matplotlib**: For basic data visualization.
- **Seaborn**: For statistical graphics.

---

### **2. Reading the Dataset**
```python
dataset = pd.read_csv("advertising.csv")
dataset.head()
```

- The dataset contains columns for TV, Radio, Newspaper, and Sales. Since we only need TV and Sales, we drop the other columns:

```python
dataset.drop(columns=['Radio', 'Newspaper'], inplace=True)
dataset.head()
```

---

### **3. Defining X and Y**
```python
X = dataset[['TV']]
Y = dataset['Sales']
```

- \( X \): Independent variable (money spent on TV advertising).
- \( Y \): Dependent variable (sales).

---

### **4. Splitting the Dataset**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
```
- **`train_test_split`** splits the data into training (70%) and testing (30%) sets.
- **`random_state`** ensures reproducibility.

---

### **5. Building the Model**
```python
from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X_train, Y_train)
```
- **`LinearRegression`** creates the model.
- **`fit()`** trains the model using the training data.

---

### **6. Model Equation**
```python
print("Intercept:", slr.intercept_)
print("Coefficient:", slr.coef_)
```
- **Output**:
  - Intercept: \( 6.948 \)
  - Coefficient: \( 0.054 \)

- **Equation**:
  \[ \text{Sales} = 6.948 + 0.054 \times \text{TV} \]

---

### **7. Making Predictions**
```python
Y_pred = slr.predict(X_test)

print("Predicted values:", Y_pred)
```
- Use the **`predict()`** method to generate predictions for the test set.

---

### **8. Comparing Actual and Predicted Values**
```python
comparison = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred})
comparison.head()
```

---

### **9. Visualizing the Line of Best Fit**
```python
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, Y_pred, color='red')
plt.title('Actual vs Predicted')
plt.xlabel('TV Advertising Spend')
plt.ylabel('Sales')
plt.show()
```

---

### **10. Model Evaluation**
```python
from sklearn import metrics

mean_ab_error = metrics.mean_absolute_error(Y_test, Y_pred)
mean_sq_error = metrics.mean_squared_error(Y_test, Y_pred)
root_mean_sq_error = np.sqrt(mean_sq_error)

print("R-squared:", slr.score(X, Y))
print("Mean Absolute Error:", mean_ab_error)
print("Mean Squared Error:", mean_sq_error)
print("Root Mean Squared Error:", root_mean_sq_error)
```

- **Evaluation Metrics**:
  - **R-squared**: Proportion of data variance explained by the model.
  - **MAE**: Average absolute difference between actual and predicted values.
  - **MSE**: Average squared difference between actual and predicted values.
  - **RMSE**: Standard deviation of prediction errors.

---

### **Conclusion**
- The Simple Linear Regression model explained **81.10%** of the variance (R-squared).
- The errors (MAE, MSE, RMSE) were low, indicating good model performance.

---

This concludes the implementation of Simple Linear Regression in Python.
