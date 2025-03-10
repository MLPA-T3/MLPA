# Regression Analysis with Python using the Advertising Dataset

This document provides a detailed guide for a class session on regression analysis using Python and the Advertising dataset. The analysis includes:

- Loading and inspecting the dataset  
- Creating a correlation heatmap  
- Generating a pairplot for all variables  
- Running a simple linear regression (predicting Sales from TV advertising) using statsmodels  
- Plotting the regression line along with interpretations

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Loading and Inspecting the Dataset](#loading-and-inspecting-the-dataset)  
3. [Correlation Heatmap](#correlation-heatmap)  
4. [Pairplot of Variables](#pairplot-of-variables)  
5. [Simple Linear Regression with Statsmodels](#simple-linear-regression-with-statsmodels)  
6. [Plotting the Regression Line](#plotting-the-regression-line)  
7. [Interpretations](#interpretations)  
8. [Conclusion](#conclusion)  

---

## Introduction

In this session, we will analyze the **Advertising** dataset which contains data on advertising expenditures (TV, Radio, Newspaper) and the corresponding **Sales**. Our focus will be on exploring the relationships between variables and performing a simple linear regression using **TV** advertising spend to predict **Sales**. We will use Python libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and `statsmodels`.

---

## Loading and Inspecting the Dataset

We start by loading the dataset and inspecting its structure and basic statistics.

    import pandas as pd

    # Load the Advertising dataset
    data = pd.read_csv("/mnt/data/Advertising.csv")
    print("First 5 rows of the dataset:")
    print(data.head())

    # Display basic information and descriptive statistics.
    print("\nDataset Information:")
    print(data.info())
    print("\nDescriptive Statistics:")
    print(data.describe())

**Explanation:**  
- **Data Loading:** The dataset is loaded into a pandas DataFrame from the CSV file.  
- **Inspection:** We print the first few rows and use `info()` and `describe()` to understand the data types, detect any missing values, and view summary statistics.

---

## Correlation Heatmap

Next, we create a correlation heatmap to visualize the linear relationships between the variables. This helps us identify which predictors might be effective.

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compute the correlation matrix
    corr_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Advertising Dataset")
    plt.show()

![image](https://github.com/user-attachments/assets/ff8ebb8d-d4c9-4b6e-8d69-81005aa176c3)

**Explanation:**  
- **Correlation Matrix:** `data.corr()` computes the Pearson correlation coefficients between pairs of variables.  
- **Heatmap:** The heatmap, with annotations and a coolwarm color scheme, visually highlights the strength and direction of the linear relationships. For example, if TV and Sales have a high positive correlation, it suggests that as TV ad spend increases, Sales tend to increase.

---

## Pairplot of Variables

A pairplot provides scatter plots for every pair of variables, along with histograms for the individual distributions. This is useful for visually assessing relationships and spotting any outliers.

    sns.pairplot(data)
    plt.suptitle("Pairplot of Advertising Dataset", y=1.02)
    plt.show()

![image](https://github.com/user-attachments/assets/ffe66946-088d-4cb3-aed0-de24affd2679)

**Explanation:**  
- **Pairplot:** Displays pairwise scatter plots for all variables along with their histograms.  
- **Super Title:** The `plt.suptitle(..., y=1.02)` positions the overall title slightly above the default to prevent overlap with the plots.

---

## Simple Linear Regression with Statsmodels

Now we perform a simple linear regression to predict **Sales** using **TV** advertising spend.

    import statsmodels.api as sm

    # Define the predictor (TV) and response (Sales) variables
    X = data["TV"]
    y = data["Sales"]

    # statsmodels requires us to add a constant term to include the intercept in the model.
    X_sm = sm.add_constant(X)

    # Create and fit the OLS model
    model = sm.OLS(y, X_sm)
    results = model.fit()

    # Print the regression summary
    print("\nStatsmodels OLS Regression Summary:")
    print(results.summary())

![Capture](https://github.com/user-attachments/assets/421e6aa5-48ff-49f5-b338-9f52c1aeb062)

**Explanation:**  
- **Predictor and Response:** We use "TV" as the independent variable and "Sales" as the dependent variable.  
- **Adding Constant:** The function `sm.add_constant(X)` adds a column of ones to account for the intercept; without it, the regression line would be forced through the origin.  
- **Fitting the Model:** The OLS (Ordinary Least Squares) method fits the regression model, and the summary provides key statistics (coefficients, p-values, R-squared, confidence intervals) to assess the model.

---

## Plotting the Regression Line

To visualize the regression results, we plot the regression line on top of a scatter plot of the data points.

    import numpy as np

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, alpha=0.6, label="Data Points")

    # Create a range of TV values for a smooth regression line
    X_range = np.linspace(X.min(), X.max(), 100)
    X_range_sm = sm.add_constant(X_range)
    y_pred = results.predict(X_range_sm)

    plt.plot(X_range, y_pred, color="red", linewidth=2, label="Regression Line")
    plt.xlabel("TV Advertising Spend")
    plt.ylabel("Sales")
    plt.title("Simple Linear Regression: Sales vs. TV Advertising")
    plt.legend()
    plt.show()

![image](https://github.com/user-attachments/assets/3a02c616-cc17-4b27-ba23-c4dad60ddc29)

**Explanation:**  
- **Scatter Plot:** Displays individual data points (TV vs. Sales) so you can see the spread of the data.  
- **Regression Line:** A red line representing the estimated relationship between TV spend and Sales, as determined by the regression model.  
- **Interpretation:** If the data points cluster closely around the regression line, it indicates that the model fits the data well.

---

# Model Summary

**Dependent Variable (Sales):** This is the outcome we want to predict.  
**R-squared (0.612):** About 61% of the variation in Sales is explained by TV advertising in this simple model. In other words, TV spending alone accounts for a little over half of the changes in Sales.  
**Adjusted R-squared (0.608):** This is very close to the regular R-squared. Since we only have one predictor (TV), there’s not much difference between the two measures.

---

# Overall Significance (F-statistic and its p-value)

**F-statistic (7.331) with p-value = 0.00753:** This tells us that, overall, the model is statistically significant—i.e., having TV in the model does a better job of explaining Sales than having no predictors at all.

---

# Coefficients

**Intercept (const = 7.0326):** When TV spending is 0, the model predicts about 7 units of Sales. This is a baseline or “starting point.”  
**TV Coefficient (0.0475):** For each 1-unit increase in TV spending, Sales are predicted to increase by about 0.0475 units on average (holding everything else constant).

---

# Statistical Significance

**p-values for Intercept and TV:** Both are very small (TV’s p-value is ~0.008, and the intercept’s p-value is practically 0). Since they are below the typical threshold of 0.05, we consider both terms to be statistically significant.

---

# Confidence Intervals

The **95% confidence interval** for the TV coefficient does not include 0 (it’s approximately [0.013, 0.082]). This reinforces that the effect of TV spending on Sales is likely positive and not zero.

---

# Interpretation 

**TV is a meaningful predictor of Sales:** As you spend more on TV advertising, you generally see higher Sales.  
**Magnitude of Effect:** A 1-unit increase in TV spending yields an estimated 0.0475 increase in Sales, on average.  
**Overall Fit:** The model explains roughly 61% of the variation in Sales, suggesting that while TV spend is important, there is still around 39% of Sales variation unaccounted for by this single predictor (other factors, like Radio, Newspaper, or market conditions, may explain the rest).

---

# Bottom Line

If you want to increase Sales, increasing TV advertising budget is likely to help. However, you’d need to consider other variables (like Radio, Newspaper, product, market trends, etc.) for a more complete explanation of Sales.

---

## In short:

- **Intercept (~7):** Predicted baseline Sales with zero TV spending.  
- **TV Coefficient (~0.05):** Expected increase in Sales per additional unit of TV spending.  
- **R-squared (~0.61):** TV alone explains about 61% of the variability in Sales.  
- **p-values (<0.05):** Both the intercept and the TV coefficient are statistically significant.



