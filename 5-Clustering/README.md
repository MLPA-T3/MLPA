# Clustering Algorithms

## What is clustering?

Clustering is a Machine Learning algorithm and a popular technique for classifying data. It falls under the category of unsupervised machine learning algorithms as it’s useful in dealing with unlabeled and unstructured data.

![image](https://github.com/user-attachments/assets/d16ddafe-6d84-4685-8bfb-f0d3b0058173)

Clustering is a great way to start making sense out of unstructured data.

In this algorithm, we usually deal with only features within data and do not have any target labels or classes. These algorithms discover hidden patterns or data groupings without the need for human intervention. Its ability to discover similarities and differences in information makes it the ideal solution for exploratory data analysis.

In other words, Clustering is a data mining technique that classifies datasets based on their similarities or differences. This process organizes raw and unclassified data objects into groups represented by structures or patterns in the information. Clustering algorithms can be categorized into a few types, specifically exclusive, overlapping, hierarchical, and probabilistic.

## Why Clustering?

To group items that might have the same attributes together. Imagine that you have millions of chemical compounds that you cannot see or analyze individually. By clustering, you can group these compounds into 5 or 10 clusters based on their similarities, making it easier to analyze them instead of examining each compound separately.

## Clustering Applications

Clustering has a wide range of applications in diverse areas such as:

- **Market research and customer segmentation**: Identify customer groups based on purchasing patterns and preferences for targeted marketing and personalized service offerings.
- **Social network analysis**: Detecting communities and understanding social dynamics.
- **Biological data analysis**: Grouping genes or proteins with similar functionalities.
- **Finance**: Grouping stocks into sectors based on various financial metrics and market behaviors.
- **Urban planning**: Grouping areas with similar land use or demographic characteristics.
- **Information retrieval**: Grouping similar documents for efficient retrieval based on user queries or interests.
- **Web search**: Search engines group similar results, enhancing user experience by categorizing information.
- **Image segmentation**: Segmenting images into meaningful clusters, such as separating objects from the background.
- **Recommender systems**: Recommending products or content by grouping similar users.
- **Anomaly detection**: Detecting unusual patterns in areas such as fraud detection and network monitoring.

## Types of Clustering

Data can be categorized under various rules and parameters. From simple similarities in data values to comparing relationships between data points, there are multiple ways to go about clustering. One way to categorize clustering techniques is as follows:

- **Partition-based Clustering**
- **Hierarchical Clustering**
- **Density-based Clustering**

We’ll briefly explain these before moving on to applications.

### Partition-Based Clustering

Given a database of *n* objects or data tuples, a partitioning method constructs *k* partitions of the data, where each partition represents a cluster.

![image](https://github.com/user-attachments/assets/b768ed54-fd19-45fa-9d5e-8911fa580c27)

This clustering method classifies information into multiple groups based on characteristics and similarities. The data analyst specifies the number of clusters for the clustering methods.

### Hierarchical Clustering

Hierarchical clustering, also known as hierarchical cluster analysis, groups similar objects into clusters. The endpoint is a set of clusters, where each cluster is distinct from others, and the objects within each cluster are broadly similar to each other.

Hierarchical clustering starts by treating each observation as a separate cluster. Then, it repeatedly executes two steps:
1. Identify the two closest clusters.
2. Merge them.

This iterative process continues until all clusters are merged into one.

Hierarchical Clustering bridges the gap created by k-means clustering by eliminating the need to predefine the number of clusters. It builds a hierarchy of clusters and can be categorized into:

- **Agglomerative Clustering**: Starts with each data point as a separate cluster and merges the closest pairs iteratively.
- **Divisive Clustering**: Starts with all data points in a single cluster and splits them iteratively.

![image](https://github.com/user-attachments/assets/15ed5ae2-fcc2-43d8-9d75-b4b8350b6ab4)

**Dendrogram:** A tree-like diagram that records the sequence of merges or splits. When clusters are merged, they are joined in the dendrogram at a height representing their distance.

![image](https://github.com/user-attachments/assets/6607851a-f211-48ee-8cd7-9f3fa32b8cf4)

### Density-Based Clustering

Density-based spatial clustering of applications with noise (DBSCAN) is a well-known data clustering algorithm used in data mining and machine learning. It groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points. Points in low-density regions are marked as outliers.

![image](https://github.com/user-attachments/assets/3ec75173-3fd7-4c0f-89bc-1579aa0e9dd2)

### Differences Between Partitioning and Hierarchical Clustering

One main difference is that in partitioning clustering, we pre-specify the number of clusters, while in hierarchical clustering, we do not.

![image](https://github.com/user-attachments/assets/d9aeff7a-8ed4-4bcd-9cd9-026fc1c095e0)

## K-Means Clustering

K-means clustering is a popular unsupervised machine learning algorithm used to group data points into *K* clusters based on similarity. The *K* in K-means represents the number of clusters that the algorithm will create.

![image](https://github.com/user-attachments/assets/b6c74ff2-2a48-4521-a3a3-3b016939a3ad)

# K-Means Clustering

The algorithm starts by selecting *K* random centroids (points in the feature space) as the initial cluster centers. It then assigns each data point to the nearest centroid based on the Euclidean distance between the data point and the centroid. Next, it computes the mean of each cluster, which becomes the new centroid for that cluster. The algorithm repeats these steps until the centroids no longer change or a maximum number of iterations is reached.

K-means clustering aims to minimize the sum of squared distances between each data point and its assigned centroid. This means that the algorithm is trying to find the cluster centroids that minimize the variance within each cluster and maximize the distance between clusters.

## How K-Means Clustering Works

K-means clustering works by iteratively assigning data points to the nearest cluster centroid, computing the mean of each cluster, and updating the centroids until convergence is reached.

![image](https://github.com/user-attachments/assets/9fb5d866-75aa-4cf4-84f0-a82f411be6de)

The following steps summarize the K-means clustering process:

1. **Initialization**: Select the number of clusters (*K*) and randomly initialize *K* cluster centroids in the feature space.
2. **Assignment**: Assign each data point to the nearest cluster centroid based on the Euclidean distance between the data point and each centroid.
3. **Update**: Compute the mean of each cluster, which becomes the new centroid for that cluster.
4. **Repeat**: Repeat steps 2 and 3 until the centroids no longer change or a maximum number of iterations is reached.
5. **Convergence**: Once convergence is reached, the *K* clusters represent groups of data points that are similar to each other and dissimilar to data points in other clusters.
6. **Visualization**: [Watch Here](https://youtu.be/2lZZ_FzlIJY)

## How Does a Computer Form Clusters?

The way the computer forms clusters is through a process of forming partitions and adjusting the position of cluster heads using well-known clustering techniques like K-Means. K-Means is a distance-based clustering technique where *K* represents the number of clusters to be formed. The goal of the K-Means is to create clusters such that the intra-cluster distance (distance of data inside the cluster) is minimized and the inter-cluster distance (distance between clusters) is maximized.

A lot of the computations used to create clusters are derived from Linear Algebra.

![image](https://github.com/user-attachments/assets/e1e8bd10-774e-4214-89ff-0254c57ec13a)

### Inter-Cluster Distance vs. Intra-Cluster Distance

Let’s walk through an example of how a computer can perform clustering with K-Means where *K = 2* clusters. Suppose a computer is given raw data about people with their weight and height. As shown in the image below, each dot represents a person, with the x-axis representing weight and the y-axis representing height. The goal is to group people into two clusters based on similar weight and height.

![image](https://github.com/user-attachments/assets/6424deff-4380-48d1-bdbe-8d7ede5c1c0e)

### Step-by-Step Process

1. **Step 1**: The computer randomly selects two data points as cluster heads.

   ![image](https://github.com/user-attachments/assets/57dcdfe3-cd58-4086-8a7c-14a34ad44fb9)

2. **Step 2**: It forms a separation in the data by finding the line that connects the two cluster heads. Then, it creates a perpendicular partition splitting the line into two.

   ![image](https://github.com/user-attachments/assets/f9a19a83-767e-42a3-b1cf-3275d62d54ec)

3. **Step 3**: Any data point left of the partition line belongs to the red cluster, while any data point right of the partition belongs to the yellow cluster.

   ![image](https://github.com/user-attachments/assets/fd344f68-730b-4ea8-9741-6650ba523635)

4. **Step 4**: The cluster head is adjusted to be the average of all points in the cluster.

   ![image](https://github.com/user-attachments/assets/032621b6-31da-4a06-b6d4-19260cbfd9fe)

5. **Step 5**: The process repeats, and a yellow point is found on the red side of the partition line.

   ![image](https://github.com/user-attachments/assets/8b123cc5-b3e0-4e2c-bc65-c43afa7f52fd)

6. **Step 6**: This yellow point is reassigned to the red cluster.

   ![image](https://github.com/user-attachments/assets/24d3b3d2-0ed3-4e59-bdd3-90da0b3f534f)

7. **Step 7**: The cluster heads are adjusted again based on the new cluster assignments.

   ![image](https://github.com/user-attachments/assets/eca3fcbb-5123-4421-93ea-5227aae77678)

8. **Step 8**: The process continues until all points are assigned to their optimal cluster.

   ![image](https://github.com/user-attachments/assets/735e284c-c2cd-4394-9cd2-4b4a200c89a0)

## Example of K-Means Clustering

Cluster the following eight points (with (x, y) representing locations) into three clusters:

**Data Points:**
- A1(2, 10)
- A2(2, 5)
- A3(8, 4)
- A4(5, 8)
- A5(7, 5)
- A6(6, 4)
- A7(1, 2)
- A8(4, 9)

**Initial Cluster Centers:**
- A1(2, 10)
- A4(5, 8)
- A7(1, 2)

**Distance Function:**

For two points *a = (x1, y1)* and *b = (x2, y2)*, the distance is defined as:

\[ \rho(a, b) = |x_2 - x_1| + |y_2 - y_1| \]

Use the K-Means Algorithm to find the three cluster centers after the second iteration.

### Iteration-01:
We calculate the distance of each point from each of the centers of the three clusters using the given distance function. The clustering process continues iteratively until the clusters stabilize.

# K-Means Clustering

## Calculating Distance Between Points

### Calculating Distance Between A1(2, 10) and C1(2, 10)

\[ \rho(A1, C1) = |x_2 - x_1| + |y_2 - y_1| \]

\[ = |2 - 2| + |10 - 10| \]

\[ = 0 \]

### Calculating Distance Between A1(2, 10) and C2(5, 8)

\[ \rho(A1, C2) = |x_2 - x_1| + |y_2 - y_1| \]

\[ = |5 - 2| + |8 - 10| \]

\[ = 3 + 2 \]

\[ = 5 \]

### Calculating Distance Between A1(2, 10) and C3(1, 2)

\[ \rho(A1, C3) = |x_2 - x_1| + |y_2 - y_1| \]

\[ = |1 - 2| + |2 - 10| \]

\[ = 1 + 8 \]

\[ = 9 \]

In a similar manner, we calculate the distance of other points from each of the centers of the three clusters.

### Next Steps

1. We draw a table showing all the results.
2. Using the table, we decide which point belongs to which cluster.
3. The given point belongs to the cluster whose center is nearest to it.

![image](https://miro.medium.com/v2/resize:fit:700/1*fgPmQ5rL03pm9DGcM7xdmQ.png)

### New Clusters

#### Cluster-01:
**Contains:**
- A1(2, 10)

#### Cluster-02:
**Contains:**
- A3(8, 4)
- A4(5, 8)
- A5(7, 5)
- A6(6, 4)
- A8(4, 9)

#### Cluster-03:
**Contains:**
- A2(2, 5)
- A7(1, 2)

### Recomputing New Cluster Centers

The new cluster center is computed by taking the mean of all points in that cluster.

#### For Cluster-01:
Since it has only one point, the center remains the same:

\[ (2, 10) \]

#### For Cluster-02:

\[ \text{Center} = \left( \frac{8 + 5 + 7 + 6 + 4}{5}, \frac{4 + 8 + 5 + 4 + 9}{5} \right) \]

\[ = (6, 6) \]

#### For Cluster-03:

\[ \text{Center} = \left( \frac{2 + 1}{2}, \frac{5 + 2}{2} \right) \]

\[ = (1.5, 3.5) \]

### Iteration-02

We calculate the distance of each point from each of the new cluster centers using the given distance function. The process continues iteratively until the clusters stabilize.

# K-Means Clustering

## Calculating Distance Between Points

### Calculating Distance Between A1(2, 10) and C1(2, 10)

\[ \rho(A1, C1) = |x_2 - x_1| + |y_2 - y_1| \]

\[ = |2 - 2| + |10 - 10| \]

\[ = 0 \]

### Calculating Distance Between A1(2, 10) and C2(6, 6)

\[ \rho(A1, C2) = |x_2 - x_1| + |y_2 - y_1| \]

\[ = |6 - 2| + |6 - 10| \]

\[ = 4 + 4 \]

\[ = 8 \]

### Calculating Distance Between A1(2, 10) and C3(1.5, 3.5)

\[ \rho(A1, C3) = |x_2 - x_1| + |y_2 - y_1| \]

\[ = |1.5 - 2| + |3.5 - 10| \]

\[ = 0.5 + 6.5 \]

\[ = 7 \]

In a similar manner, we calculate the distance of other points from each of the centers of the three clusters.

### Next Steps

1. We draw a table showing all the results.
2. Using the table, we decide which point belongs to which cluster.
3. The given point belongs to the cluster whose center is nearest to it.

![image](https://github.com/user-attachments/assets/c5fa5c07-d234-469e-b4e9-6f90f728a51c)

### New Clusters

#### Cluster-01:
**Contains:**
- A1(2, 10)
- A8(4, 9)

#### Cluster-02:
**Contains:**
- A3(8, 4)
- A4(5, 8)
- A5(7, 5)
- A6(6, 4)

#### Cluster-03:
**Contains:**
- A2(2, 5)
- A7(1, 2)

### Recomputing New Cluster Centers

The new cluster center is computed by taking the mean of all points in that cluster.

#### For Cluster-01:

\[ \text{Center} = \left( \frac{2 + 4}{2}, \frac{10 + 9}{2} \right) \]

\[ = (3, 9.5) \]

#### For Cluster-02:

\[ \text{Center} = \left( \frac{8 + 5 + 7 + 6}{4}, \frac{4 + 8 + 5 + 4}{4} \right) \]

\[ = (6.5, 5.25) \]

#### For Cluster-03:

\[ \text{Center} = \left( \frac{2 + 1}{2}, \frac{5 + 2}{2} \right) \]

\[ = (1.5, 3.5) \]

### Iteration-02

After the second iteration, the center of the three clusters is:

- **C1:** (3, 9.5)
- **C2:** (6.5, 5.25)
- **C3:** (1.5, 3.5)

## How to Choose the Optimal Value of K

The optimal value of *K* can be chosen using the **elbow method**. In this method, the cost function (variance) is plotted against different values of *K*. As the value of *K* increases, the number of clusters decreases and the average distortion also decreases. The optimal value of *K* is identified where we see a sharp change in the rate of decrease in error, forming an "elbow" in the plot.

For example, in the following illustration, the optimal value of *K* is **3**.

![image](https://github.com/user-attachments/assets/f1c11a0f-1efa-430e-9eca-6c72cd4cff4f)

# Hierarchical Clustering

Hierarchical clustering is a type of clustering algorithm used in machine learning and data analysis to group similar objects into clusters. It works by iteratively merging the closest pairs of data points or clusters until all data points are in a single cluster or until a desired number of clusters is reached.

## Types of Hierarchical Clustering

There are two types of hierarchical clustering:

- **Agglomerative**: Starts with individual data points as clusters and merges them together iteratively.
- **Divisive**: Starts with all data points in a single cluster and recursively divides them into smaller clusters.

Hierarchical clustering is useful in applications such as image segmentation, social network analysis, and gene expression analysis.

## How Hierarchical Clustering Works

Hierarchical clustering is a clustering algorithm that works by iteratively merging or dividing clusters of data points based on their similarity. The algorithm can be performed in two ways: **agglomerative** and **divisive**.

### Agglomerative Hierarchical Clustering

Agglomerative hierarchical clustering begins with each data point as a separate cluster and iteratively merges the closest pairs of clusters until all data points belong to a single cluster or a desired number of clusters is reached.

![image](https://github.com/user-attachments/assets/f001c197-4a61-4f66-bfee-fdb0a82668b7)

The steps involved in agglomerative hierarchical clustering are:

1. Calculate the distance matrix between all pairs of data points.
2. Assign each data point to its own cluster.
3. Find the pair of clusters with the smallest distance and merge them into a single cluster.
4. Update the distance matrix by computing the distances between the new cluster and all other clusters.
5. Repeat steps 3 and 4 until all data points belong to a single cluster or the desired number of clusters is reached.

### Divisive Hierarchical Clustering

Divisive hierarchical clustering begins with all data points in a single cluster and recursively divides them into smaller clusters until each data point belongs to its own cluster or a desired number of clusters is reached.

![image](https://github.com/user-attachments/assets/529c3844-94ef-4743-a847-55383de0aa72)

The steps involved in divisive hierarchical clustering are:

1. Assign all data points to a single cluster.
2. Compute the distance matrix between all pairs of data points.
3. Find the data point that is farthest away from all other data points and assign it to a new cluster.
4. Divide the remaining data points into two clusters based on their distance from the newly created cluster.
5. Repeat step 4 recursively until each data point belongs to its own cluster or the desired number of clusters is reached.

### Linkage Criteria in Hierarchical Clustering

The choice of distance metric and linkage criteria, which define how distances between clusters are computed, can have a significant impact on the clustering results. Common distance metrics include **Euclidean distance, Manhattan distance,** and **cosine distance,** while linkage criteria include **single linkage, complete linkage,** and **average linkage.**

#### Single Linkage

In single linkage hierarchical clustering, the distance between two clusters is defined as the shortest distance between two points in each cluster.

![image](https://github.com/user-attachments/assets/70e11db4-2999-4c67-a72f-f097daeff752)

#### Complete Linkage

In complete linkage hierarchical clustering, the distance between two clusters is defined as the longest distance between two points in each cluster.

![image](https://github.com/user-attachments/assets/b5f09d00-be5d-444c-9ab7-4283062b2e62)

#### Average Linkage

In average linkage hierarchical clustering, the distance between two clusters is defined as the average distance between each point in one cluster to every point in the other cluster.

![image](https://github.com/user-attachments/assets/294aed3c-b138-4ec8-9f2a-3b9c1f132e6e)

## Example: Performing Hierarchical Clustering

### Objective

For the one-dimensional data set \{7, 10, 20, 28, 35\}, perform hierarchical clustering and plot the dendrogram to visualize it.

### Solution

First, let’s visualize the data:

![image](https://github.com/user-attachments/assets/98b4b9bd-6fc2-4465-a0a1-735b8050a8d6)

Observing the plot above, we can intuitively conclude that:

- The first two points \(7\) and \(10\) are close to each other and should be in the same cluster.
- The last two points \(28\) and \(35\) are close to each other and should be in the same cluster.
- The cluster of the center point \(20\) is not easy to conclude.

Let’s solve the problem by hand using agglomerative hierarchical clustering:

### Single Linkage

In single linkage hierarchical clustering, we merge in each step the two clusters whose two closest members have the smallest distance.

![image](https://github.com/user-attachments/assets/29a9f279-e99a-4fc8-ac96-4d24a93b43e5)

Using single linkage, two clusters are formed:

- **Cluster 1**: \(7,10\)
- **Cluster 2**: \(20,28,35\)

### Example of Hierarchical Clustering with More Data Points

Clustering the following 7 data points:

![image](https://github.com/user-attachments/assets/17fd9d5d-d49f-4f62-805e-5fcc8e4c2550)

![image](https://github.com/user-attachments/assets/387d22ce-7167-48a6-9dfd-37a639a8ee42)

The vertical axis represents the distance between the clusters, while the horizontal axis represents the data points or clusters being merged or divided.

### Steps in Hierarchical Clustering Example

#### Step 1
Calculate distances between all data points using the Euclidean distance function. The shortest distance is between data points C and G.

![image](https://github.com/user-attachments/assets/fc9abc9b-7f45-4471-b131-4358f3a6d6a4)

#### Step 2
We use **Average Linkage** to measure the distance between the **C, G** cluster and other data points.

![image](https://github.com/user-attachments/assets/bac023c5-3d48-4b18-8816-0ad92b657994)

# Hierarchical Clustering

Hierarchical clustering is a type of clustering algorithm used in machine learning and data analysis to group similar objects into clusters. It works by iteratively merging the closest pairs of data points or clusters until all data points are in a single cluster or until a desired number of clusters is reached.

## Types of Hierarchical Clustering

There are two types of hierarchical clustering:

- **Agglomerative**: Starts with individual data points as clusters and merges them together iteratively.
- **Divisive**: Starts with all data points in a single cluster and recursively divides them into smaller clusters.

Hierarchical clustering is useful in applications such as image segmentation, social network analysis, and gene expression analysis.

## How Hierarchical Clustering Works

Hierarchical clustering is a clustering algorithm that works by iteratively merging or dividing clusters of data points based on their similarity. The algorithm can be performed in two ways: **agglomerative** and **divisive**.

### Agglomerative Hierarchical Clustering

Agglomerative hierarchical clustering begins with each data point as a separate cluster and iteratively merges the closest pairs of clusters until all data points belong to a single cluster or a desired number of clusters is reached.

![image](https://github.com/user-attachments/assets/f001c197-4a61-4f66-bfee-fdb0a82668b7)

The steps involved in agglomerative hierarchical clustering are:

1. Calculate the distance matrix between all pairs of data points.
2. Assign each data point to its own cluster.
3. Find the pair of clusters with the smallest distance and merge them into a single cluster.
4. Update the distance matrix by computing the distances between the new cluster and all other clusters.
5. Repeat steps 3 and 4 until all data points belong to a single cluster or the desired number of clusters is reached.

### Divisive Hierarchical Clustering

Divisive hierarchical clustering begins with all data points in a single cluster and recursively divides them into smaller clusters until each data point belongs to its own cluster or a desired number of clusters is reached.

![image](https://github.com/user-attachments/assets/529c3844-94ef-4743-a847-55383de0aa72)

The steps involved in divisive hierarchical clustering are:

1. Assign all data points to a single cluster.
2. Compute the distance matrix between all pairs of data points.
3. Find the data point that is farthest away from all other data points and assign it to a new cluster.
4. Divide the remaining data points into two clusters based on their distance from the newly created cluster.
5. Repeat step 4 recursively until each data point belongs to its own cluster or the desired number of clusters is reached.

### Steps in Hierarchical Clustering Example

#### Step 3: Continue with Step 2

![image](https://github.com/user-attachments/assets/1524d059-d111-4b55-a078-d9f0f3f9b11c)

#### Step 4: Continue with Step 3

![image](https://github.com/user-attachments/assets/f18b581a-202c-4144-a26f-5644e0b2ed0b)

#### Step 5: Continue with Step 4

![image](https://github.com/user-attachments/assets/ec5c1b9a-eb3a-4ad6-97fb-491ad11ae8e7)

#### Step 6: Continue with Step 5

![image](https://github.com/user-attachments/assets/e2a02ff4-0e08-4953-8b12-01e8bd76d5a0)

### Final Hierarchical Clustering Result

A diagram to show the results of hierarchical clustering applied to a dataset with 10 data points:

![image](https://github.com/user-attachments/assets/05e46a99-d240-4fd6-aa1e-e257c1b8fbbf)

## Feature Selection for Clustering Algorithms

Feature selection is crucial in clustering algorithms. Aside from choosing only features that will be available when the model is deployed, you should only pick features that you want your clusters to be differentiated by, either for a business objective or an analytical reason.

Before selection, perform a **correlation analysis** to avoid redundancy—features that seem unrelated might actually be correlated, and keeping both may not add value.

For instance, in a **bank customer segmentation** task, including location as a clustering feature might be irrelevant if you aim for a homogeneous marketing strategy across regions. The clustering model may separate customers based on location instead of meaningful financial behavior.

## Feature Scaling

Feature scaling standardizes the range of data features to ensure they contribute equally to clustering. Without scaling, features with large numerical ranges dominate distance calculations, biasing cluster formation.

Example:
- **Bank balance**: 0–1,000,000 USD
- **Age**: 18–100 years

Without scaling, the bank balance would dominate distance computations. Scaling techniques like normalization (0-1 range) or standardization (mean=0, std=1) bring features to comparable ranges.

**Effect on Clustering Models:**

K-means clustering on **Weight (40–130kg) and Height (1.4–2m)**:

**Without Scaling:**

![image](https://github.com/user-attachments/assets/85f58f18-7fd5-40eb-839e-eab21ea215f4)

Clusters are formed based on weight alone, ignoring height.

**With Scaling:**

![image](https://github.com/user-attachments/assets/aba9c599-9859-4779-b074-8014001e8a97)

Clusters now consider both weight and height equally.

## Dimensionality Reduction

Dimensionality reduction techniques reduce dataset dimensions while retaining as much information as possible. 

A key method is **Principal Component Analysis (PCA)**, often used before clustering to counteract the **curse of dimensionality**, where increasing dimensions blur meaningful distances.

**Effect of High Dimensions:**

As dimensions increase, data points become more spread out, affecting clustering distance metrics.

![image](https://github.com/user-attachments/assets/e4f0b194-f5d7-445c-a5da-d1ef99b0822b)

To ensure effective clustering:
- Use **feature selection** to retain only key attributes.
- Apply **PCA** or other techniques to project data onto a lower-dimensional space while preserving information.

Once **feature selection, scaling, and dimensionality reduction** are complete, clustering can be performed more effectively.


