# Student Marks Prediction Using Machine Learning

**Student marks prediction** is a popular data science case study based on the problem of regression. It is an excellent starting point for data science beginners as it is easy to solve and understand. This guide will demonstrate how to predict student marks using machine learning and Python.

---

## Overview of the Problem

You are given information about students, including:

- **The number of courses** they have opted for.
- **The average time studied** per day.
- **Marks obtained.**

The objective is to predict the marks of other students using this data.

---

## Data Description

The dataset contains three columns:

- **`number_courses`**: Number of courses taken by the student.
- **`time_study`**: Average time studied daily by the student.
- **`Marks`**: Marks scored by the student (target column).

### Loading the Libraries and Dataset

Before starting, import the necessary libraries and load the dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("Student_Marks.csv")
print(data.head(10))
```

Sample output:

```
   number_courses  time_study   Marks
0               3       4.508  19.202
1               4       0.096   7.734
2               4       3.133  13.811
3               6       7.909  53.018
4               8       7.811  55.299
```

### Checking for Missing Values

Before proceeding, we need to check if the dataset contains any null values:

```python
print(data.isnull().sum())
```

The dataset has no null values and is ready for use.

---

## Analyzing the Data

### Number of Courses Distribution

To understand the distribution of courses:

```python
data["number_courses"].value_counts()
```

### Visualization of Number of Courses vs Marks Scored

To analyze the relationship between the number of courses and marks scored, use a scatter plot:

```python
plt.scatter(data['number_courses'], data['Marks'], s=data['time_study']*10, alpha=0.6)
plt.title("Number of Courses vs Marks Scored")
plt.xlabel("Number of Courses")
plt.ylabel("Marks Scored")
plt.grid(True)
plt.show()
```
![Courses Image](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2022/04/courses.png?w=700&ssl=1)

### Analyzing Time Spent vs Marks Scored

To analyze the relationship between the time spent studying and marks scored:

```python
plt.scatter(data['time_study'], data['Marks'], s=data['number_courses']*10, alpha=0.6, c='blue')
plt.title("Time Spent vs Marks Scored")
plt.xlabel("Time Spent (hours)")
plt.ylabel("Marks Scored")
plt.grid(True)
plt.show()
```

![Time Image](https://i0.wp.com/thecleverprogrammer.com/wp-content/uploads/2022/04/time.png?w=700&ssl=1)


### Correlation Analysis

Find the correlation between the features and the target column:

```python
correlation = data.corr()
print(correlation["Marks"].sort_values(ascending=False))
```

---

## Building the Prediction Model

### Splitting the Data

Split the dataset into training and testing sets:

```python
x = data[["time_study", "number_courses"]]
y = data["Marks"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Training the Model

Train a linear regression model:

```python
model = LinearRegression()
model.fit(xtrain, ytrain)
print("Model Score:", model.score(xtest, ytest))
```

### Testing the Model

Predict the marks of a student:

```python
features = [[4.508, 3]]
predicted_marks = model.predict(features)
print("Predicted Marks:", predicted_marks)
```
Predicted Marks: [22.30738483]
---


### Number of Courses vs Marks

The scatter plot above suggests that the number of courses a student takes might not directly affect their marks. However, students studying for more time tend to perform better.

### Time Spent Studying

The analysis indicates a linear relationship between the time studied and the marks obtained. This implies that spending more time studying can significantly improve performance.

### Correlation Analysis

The `time_study` column shows a strong positive correlation with marks, making it a key predictor in the dataset.

---

## Summary

This guide demonstrated how to solve the problem of student marks prediction using machine learning. Here’s a quick recap:

1. **Data Analysis:** We explored the relationships between features like time studied, number of courses, and marks scored.
2. **Model Building:** We trained a linear regression model to predict student marks.
3. **Prediction:** The model can predict marks based on the number of courses and time spent studying.

This regression problem helps beginners understand how to handle datasets, visualize data, and build predictive models effectively. Feel free to extend this problem with additional features or explore other regression models for better accuracy.
