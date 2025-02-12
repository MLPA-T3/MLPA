
# Comprehensive Guide to Pandas Using the Titanic Dataset

## Introduction
This guide will introduce you to Pandas using the Titanic dataset. By the end of this guide, you will be comfortable with loading data, performing data analysis, handling missing data, grouping, creating pivot tables, feature engineering, and visualizing data using Pandas.

---

## Step 1: Loading the Titanic Dataset

First, we need to load the dataset using Pandas.
```python
import pandas as pd

# Load the dataset
df = pd.read_csv("titanic.csv")

# Display the first 5 rows of the dataset
df.head()
```
### Understanding the Dataset Structure
To understand the dataset, we need to check its structure and summary.
```python
# Check dataset structure and data types
df.info()

# Display basic statistics for numeric columns
df.describe()

# Check for missing values
df.isnull().sum()
```
**Explanation:**
- `df.info()` shows the column names, data types, and non-null values.
- `df.describe()` gives statistics like mean, standard deviation, and percentiles for numeric columns.
- `df.isnull().sum()` helps identify missing values in each column.

---

## Step 2: Data Selection and Slicing

### Selecting Columns
```python
# Select a single column (returns a Series)
df['Age']

# Select multiple columns (returns a DataFrame)
df[['Age', 'Sex', 'Survived']]
```
### Slicing Rows
```python
# Select the first 10 rows
df[:10]

# Use loc and iloc
# loc: label-based indexing
df.loc[0:5, ['Name', 'Age', 'Sex']]

# iloc: integer-based indexing
df.iloc[0:5, 0:3]
```
### Conditional Selection
```python
# Select passengers older than 30
df[df['Age'] > 30]

# Select female passengers in first class
df[(df['Sex'] == 'female') & (df['Pclass'] == 1)]

# Passengers who embarked from Cherbourg and are younger than 20
df[(df['Embarked'] == 'C') & (df['Age'] < 20)]
```
---

## Step 3: Handling Missing Data

Handling missing data is crucial for any data analysis.
```python
# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop rows with missing Embarked values
df.dropna(subset=['Embarked'], inplace=True)
```
### Advanced Handling Techniques
```python
# Fill missing values in the Cabin column with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

# Interpolate missing Age values
df['Age'] = df['Age'].interpolate()
```
---

## Step 4: Grouping and Aggregation

Grouping and aggregation allow you to summarize the data.
```python
# Count of survivors by class
df.groupby('Pclass')['Survived'].sum()

# Average age by class and gender
df.groupby(['Pclass', 'Sex'])['Age'].mean()

# Maximum fare paid by each class
df.groupby('Pclass')['Fare'].max()
```
### Multi-level Grouping
```python
# Survival rate by class and embarkation point
df.groupby(['Pclass', 'Embarked'])['Survived'].mean()
```
---

## Step 5: Creating Pivot Tables
Pivot tables help summarize data in a tabular format.
```python
# Survival rate by class and gender
df.pivot_table(index='Pclass', columns='Sex', values='Survived', aggfunc='mean')

# Count of passengers by Embarked and Pclass
df.pivot_table(index='Embarked', columns='Pclass', values='Name', aggfunc='count')
```
### Advanced Pivot Table Examples
```python
# Survival rate by class, gender, and embarkation point
df.pivot_table(index='Pclass', columns=['Sex', 'Embarked'], values='Survived', aggfunc='mean')

# Average fare by class and survival status
df.pivot_table(index='Pclass', columns='Survived', values='Fare', aggfunc='mean')
```
---

## Step 6: Data Visualization
We can visualize the data using Matplotlib and Seaborn.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot survival count by class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

# Plot distribution of Age
sns.histplot(df['Age'], bins=20, kde=True)
plt.show()
```
### Advanced Visualization
```python
# Boxplot for Fare by Class and Survival status
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=df)
plt.show()

# Pairplot for numerical features
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived')
plt.show()
```
---

## Step 7: Feature Engineering
Create new columns or modify existing ones to enhance the dataset.
```python
# Create a new column indicating if the passenger is a child
df['Is_Child'] = df['Age'] < 18

# Create a new column for family size
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Create a new column indicating if the passenger was traveling alone
df['Is_Alone'] = (df['Family_Size'] == 1)
```
### Binning and Encoding
```python
# Binning Age into categories
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 120], labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior'])

# One-hot encoding for Embarked column
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
```
---

## Step 8: Exporting the Cleaned Dataset
Finally, save the cleaned dataset to a new CSV file.
```python
# Save the cleaned dataset
df.to_csv('cleaned_titanic.csv', index=False)
```
---

## Conclusion
This guide covers the essential steps for analyzing data using Pandas with the Titanic dataset. You should now be able to:
- Load and explore datasets
- Handle missing data
- Group and summarize data
- Create pivot tables
- Perform feature engineering
- Visualize data

### Suggested Exercises
1. Find the average fare paid by survivors vs. non-survivors.
2. Create a pivot table showing survival rate by family size.
3. Visualize the distribution of fares paid using a histogram.
4. Engineer a new feature to classify passengers as "High Fare" or "Low Fare" based on the median fare.
