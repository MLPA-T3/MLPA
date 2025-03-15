## Customer Segmentation Using K-Means Clustering

Customer segmentation is simply grouping customers with similar characteristics. These characteristics include geography, demography, behavioural, purchasing power, situational factors, personality, lifestyle, psychographic, etc. The goals of customer segmentation are customer acquisition, customer retention, increasing customer profitability, customer satisfaction, resource allocation by designing marketing measures or programs and improving target marketing measures \[1\].

Clustering is an efficient technique used for customer segmentation. Clustering places homogenous data points in a given dataset. Each of these groups is called a cluster \[2\]. While the objects in each cluster are similar between themselves, they are dissimilar to the objects of other groups. Clustering is a type of data mining approach in machine learning classified under unsupervised learning \[3\]. This is because it is able to discover patterns and information from unlabelled data. It is used extensively in machine learning, classification, and pattern recognition \[3\].

Clustering algorithms include the K-means algorithm, hierarchical clustering, DBSCAN \[4\]. In this project, the k-means clustering algorithm has been applied in customer segmentation. K-means is a clustering algorithm based on the principle of partition \[5\]. The letter k represents the number of clusters chosen. It is the most common centroid-based algorithm.

The steps of K-means clustering are \[5\]:

1\. Determine the number of clusters (k).  
2\. Select initial centroids.  
3\. Map each data point into the nearest cluster (most similar to centroid).  
4\. Update the mean value (centroid) of each cluster.  
5\. Repeat step 3–4 until all centroids are not changed.

Let’s jump right in!

## **_Problem Statement_**

A retail store wants to get insights about its customers. And then build a system that can cluster customers into different groups.

## **_About the dataset_**

The dataset can be downloaded from the kaggle website which can be found [here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

The data includes the following features:

1\. Customer ID  
2\. Customer Gender  
3\. Customer Age  
4\. Annual Income of the customer (in Thousand Dollars)  
5\. Spending score of the customer (based on customer behaviour and spending nature)

**Import libraries**

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
```

**Read dataset**

```python
df= pd.read_csv('/content/Customers.csv', index_col = 0) #loads the csv file into a pandas dataframe
df.head() #returns the first 5 rows
```


![](https://miro.medium.com/v2/resize:fit:663/1*BgXiZ150O3yxTFf19CSBOQ.png)

First five rows of the dataset. Image by the author.

## Exploratory Data Analysis (EDA)

In this step, we will perform the operations below to check what the data set comprises of. We will do the following:

— renaming a column in the dataset  
— checking data types  
— Descriptive statistics  
— Looking for null or missing values  
— Looking for duplicated values

## **Renaming a column in the dataset**

The rename() function is used to the columns in a dataframe. we renamed the second column from ‘Genre’ to ‘Gender’

```python
df.rename(columns= {'Genre': 'Gender'}, inplace = True) #To rename column 2 from Genre to Gender
df.head() #Checking if the correction has been effected
```

![](https://miro.medium.com/v2/resize:fit:664/1*mJVm8IfRAa61PKDqfYTBtA.png)

df.rename( ). Image by the author

## Checking data types and shape

The dtypes function is used to check what all the data types are in a dataframe.

```python
df.dtypes #returns the data types of the variables
```

![](https://miro.medium.com/v2/resize:fit:414/1*BqzjUB9w8pzfEkKXUItC8A.png)

df.dtypes( ). Image by the author

The shape function is used to check the number of rows and columns.

```python
df.shape #retuns the number of rows and columns in the dataset.
```

There are 200 rows (data points) and 4 columns in our dataset.

## Descriptive statistics

The describe() function is used to get a descriptive statistics summary of a given dataframe. This includes mean, count, std deviation, percentiles, and min-max values of all the features.

```python
df.describe() #returns the descriptive statistics of the dataset.
```

![](https://miro.medium.com/v2/resize:fit:615/1*JXqTZq3kgxKZLYO4hmCeMQ.png)

Descriptive statistics summary of the dataframe. Image by the author.

## Looking for null or missing values

The isnull() function is used to detect missing values.

```python
df.isnull().sum() #returns the number of missing values
```

![](https://miro.medium.com/v2/resize:fit:623/1*-HwwMbjOWyI2hPnIITYA9g.png)

isnull( ) to check for missing values. Image by the author

## Looking for duplicated values

The duplicated() function is used to find duplicate rows in a dataframe. It returns a Series with True and False values that describe which rows in the dataframe are duplicated and not.

```python
df.duplicated() #Checking for duplicate values.
```

![](https://miro.medium.com/v2/resize:fit:372/1*jyVHhKZGDDYoJDcJzMSANw.png)

df.duplicated( ). Image by the author.

## Bivariate Analysis — Scatterplot

We are interested in identifying the relationship between the Annual Income (k$) and Spending Score (1–100) we would use the scatterplot.

```python
sns.set_style('dark')
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = df)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatterplot Between Annual Income (k$) and Spending Score (1-100)')
plt.show()
```
![](https://miro.medium.com/v2/resize:fit:700/1*QwEFxFHzTlLvGGUKSOKLwA.png)

Scatterplot between Annual Income (k$) and Spending Score (1–100). Image by the author.

## Feature Selection(Choosing the columns of interest for clustering)

We are only interested in the Annual Income (k$) and Spending Score (1–100). So let’s extract these columns from our dataset using the .loc() function.

```python
X = df.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values
```

## Feature Normalization

Feature normalization helps to adjust all the data elements to a common scale in order to improve the performance of the clustering algorithm. For example in our data set Annual Income is having values in thousands and spending score in just two digits. Since the data in these variables are of different scales, it is tough to compare these variables. Each data point is converted to the range of 0 to +1. Normalization techniques include Min-max, decimal scaling and z-score. The MinMaxScaler normalization technique was used to normalize the features before running the k-Means algorithm on the dataset.

```python
scaler = MinMaxScaler().fit(X) #It makes an object of the MinMaxScaler and then we fit it on our variable X. 
print(scaler)
```

Feature normalization. 
```python
scaler.feature_range
```
```python
scaler.transform(X) 
```
```
array([[0.        , 0.3877551 ],
       [0.        , 0.81632653],
       [0.00819672, 0.05102041],
       [0.00819672, 0.7755102 ],    
       [0.01639344, 0.39795918],
```
       
The output shows the transformed numbers of our variables Annual Income (k$) and Spending Score (1–100). All the values are between 0 and 1. There are no negative values and no number is greater than 1.

## Choosing the Optimum Number of Clusters

a) To find the optimum number of clusters we’d used the WCSS (Within Clusters Sum of Squares)

**WCSS** is defined as the sum of the squared distance between each member of the cluster and its centroid. We create a for loop to find the wcss value when we consider one cluster and then two clusters up to 10. And then find the minimum wcss value

```python
wcss = []

for i in range(1,11):
    kmeans= KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X) #Finds the Kmeans to our data
    wcss.append(kmeans.inertia_) #This gives us the wcss values for each clusters
```

The “init” argument is the method for initializing the centroid.

b)Plot an Elbow graph

The **elbow graph** is used in determining the number of clusters in a data set.

```python
plt.figure(figsize = (12,6))
plt.grid()
plt.plot(range(1,11),wcss, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show
```

![](https://miro.medium.com/v2/resize:fit:700/1*gssZlB32pXs8AZBpITOs4Q.png)

Elbow graph. Image by the author

From the above graph we can observe that between number of cluster = 4 to number of cluster = 6 there has been substantial decrease(an elbow) hence, we chose the K value for our dataset as 5.

## Training the K-Means Clustering Model

Now let’s train the model on the dataset with a number of clusters 5.

```python
kmeans= KMeans(n_clusters = 5, init = 'k-means++') #initialize the class object
label= kmeans.fit_predict(X) #returns a cluster number for each of the data points
print(label)
```

![](https://miro.medium.com/v2/resize:fit:700/1*tbG8ixVUjfGBlMkdOdrM9w.png)

cluster number for each of the data points. Image by the author.

The values above represent a cluster number for each of the data points.

## Checking the centers of out clusters (Also known as Centroids)

```python
print(kmeans.cluster_centers_)
```

![](https://miro.medium.com/v2/resize:fit:700/1*ZWsD6MCqW4WOPCOqbVHSfQ.png)

X and Y centroid coordinates of all the clusters that have been created. Image by the author.

## Visualizing all the clusters

```python
plt.figure(figsize=(8,8))
plt.scatter(X[label == 0,0], X[label== 0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[label == 1,0], X[label== 1,1], s=50, c='yellow', label='Cluster 2')
plt.scatter(X[label == 2,0], X[label== 2,1], s=50, c='red', label='Cluster 3')
plt.scatter(X[label == 3,0], X[label== 3,1], s=50, c='purple', label='Cluster 4')
plt.scatter(X[label == 4,0], X[label== 4,1], s=50, c='blue', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_ [:,0], kmeans.cluster_centers_ [:,1], s= 100, c='black', marker= '*', label='Centriods') #Plotting the centriods
plt.title('Customer groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

![](https://miro.medium.com/v2/resize:fit:497/1*4BIY-fskwoziqbQj3NsssQ.png)

clusters. Image by the author.

## Business Insights

The result of the analysis shows that the retail store customers can be group into 5 clusters or segments for targeted marketing.

**Cluster 1 (green):** These are average income earners with average spending scores. They are cautious with their spending at the store.

**Cluster 2 (yellow):** The customers in this group are high income earners and with high spending scores. They bring in profit. Discounts and other offers targeted at this group will increase their spending score and maximize profit.

**Cluster 3 (red):** This group of customers have a higher income but they do not spend more at the store. One of the assumption could be that they are not satisfied with the services rendered at the store. They are another ideal group to be targeted by the marketing team because they have the potential to bring in increased profit for the store.

**Cluster 4 (purple):** Low income earners with low spending score. I can assume that this is so because people with low income will tend to purchase less item at the store.

**Cluster 5 (blue)**: These are low income earning customers with high spending scores. I can assume that why this group of customers spend more at the retail store despite earning less is because they enjoy and are satisfied with the services rendered at the retail store.

## References

1\. Sausen, K., Tomczak, T., & Herrmann, A. (2005). Development of a taxonomy of strategic market segmentation: a framework for bridging the implementation gap between normative segmentation and business practice. _Journal of Strategic Marketing_, _13_(3), 151–173.

2\. Kashwan, K. R., & Velu, C. M. (2013). Customer segmentation using clustering and data mining techniques. _International Journal of Computer Theory and Engineering_, _5_(6), 856.

3\. Rai, P., & Singh, S. (2010). A survey of clustering techniques. _International Journal of Computer Applications_, _7_(12), 1–5.

4\. T.Nelson Gnanaraj, Dr.K.Ramesh Kumar N.Monica. ―Survey on mining clusters using new k-mean algorithm from structured and unstructured data‖. International Journal of Advances in Computer Science and Technology. 2007. Volume 3, №2.

5\. Han, J., Pei, J., & Kamber, M. (2011). _Data mining: concepts and techniques_. Elsevier.
