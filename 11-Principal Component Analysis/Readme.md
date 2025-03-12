## Prinicpal Component Analysis

In a modern data analysis world, **_Principal Component Analysis (PCA)_** is a technique commonly used in a variety of fields for **_image compressions_**, **_noise reductions_**, **_dimensionality reductions_**, **_Exploratory_** **_data analysis,_** and more., by reducing low important features in the dataset.

For example, PCA removes noise (irrelevant features) in the data and keeps more relevant important features in a dataset. This helps to solve the overfitting of the data and increases the model performance.

> There is a fundamental question! How Principal Component Works (PCA) works!!!

To explain PCA, we need to understand some basic statistical concepts.

-   MEAN
-   MEDIAN
-   STANDARD DEVIATION

![](https://miro.medium.com/v2/resize:fit:437/1*DHuYRzHunJ_uyKnwpmwsHw.png)

Image Credits: Andertoons by Mark Anderson

Let us use a small dataset to identify the mean, median, and standard deviation.

**_Data Set:_**

> We ask five people how many hours of cellphones they use in a week.

![](https://miro.medium.com/v2/resize:fit:442/1*4Zei0Ptac_RDgM2oa4v0xw.png)

![](https://miro.medium.com/v2/resize:fit:477/1*0ZMvY3tMOnn9HOeMMNxOEw.png)

## **Mean**

> Mean is finding the average of the given numbers.

![](https://miro.medium.com/v2/resize:fit:628/1*mbMKM4kZ6LIeCqiCcW9TSA.png)

Hence, on average 12 hours, they use their cellphones a week.

## Median

> Midpoint of the data

-   Arrange the data from the smallest to the largest
-   Choose the value in the middle.

![](https://miro.medium.com/v2/resize:fit:146/1*2LMbsxv8SvwylehXUBDtsA.png)

The above-rounded value is the median!

**Standard Deviation**

> Standard Deviation (SD), tells us the spread of data

-   It describes the data measure dispersion
-   Standard Deviation (SD) measures the spread from the mean

![](https://miro.medium.com/v2/resize:fit:341/1*HNhKbjGRPslYDTccvrALIA.png)

We already know the mean value for the given dataset.

![](https://miro.medium.com/v2/resize:fit:472/1*i3wp-t0ce3FY6hiQUMHQVg.png)

Hence, **_mean = 12._**

![](https://miro.medium.com/v2/resize:fit:700/1*bQY9ol05jFJkfurkazYGNg.png)

As we know, Standard Deviation (SD) tells us, how data is spread from the mean.

-   _Low SD means data are clustered around the mean_
-   _High SD means indicates data are more spread out!_

_Let us see where and why we use Mean, Standard Deviation (SD) in PCA!!!_

> In Supervised learning, we know the label or target to predict a class or a feature. In Unsupervised learning, we do not know which labels or features are important. Therefore, **_Principal Component Analysis (PCA)_** helps to identify the important features or labels before applying any ML algorithms to predict or forecast features.

![](https://miro.medium.com/v2/resize:fit:482/1*YIMJdkB68rOA5ZOgMQGAmw.png)

Image Source: [https://bit.ly/3RlxlDw.](https://bit.ly/3RlxlDw.) In Unsupervised learning, when you do not know what are all the important features.

Now, _How does PCA tell us which features are important?_

**_For this PCA exploratory data analysis use case, let us take Online Shoppers Purchasing Intention Dataset from the_** [**_UCI_**](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) **_machine learning repository._**

The dataset consists of information about online shoppers page click analytics some of the columns include,

-   **ProductRelated**: The dataset default is updated with a label encoder. We will assume, 1 — Apple iPhone, 2 — Reebok Shoes, 3 — Kids Toy, and so on.
-   **ProductRelated\_Duration**: How much time each customer spent time on the particular product page
-   **BounceRates:** feature for a web page refers to the percentage of visitors who enter the site from that page and then leave (“bounce”) without triggering any other requests to the analytics server during that session
-   **ExitRates:** feature for a specific web page is calculated as for all pageviews to the page, the percentage that was the last in the session.
-   **SpecialDay**: Valentine's Day, Mother’s Day
-   **Month**: Feb, March
-   **VistitorType**: Returning Visitor, New Visitor
-   **Weekend**: True, False.

![](https://miro.medium.com/v2/resize:fit:700/1*g-Ljr8J6ArpLpMZjifmL6g.png)

**Image:** _Online Shoppers Purchasing Intention Dataset. You can refer to & download this dataset as CSV from my GitHub link_

> By using this dataset, our job is to do exploratory analysis using PCA to identify why last year revenue was reduced and how we can improve it

**_Below is the set of procedures and steps that PCA mathematically follows to help us to identify important features or labels._**

1.  Standardize the data set
2.  Find Covariance
3.  Identify Principal Components (PCs) by calculating Eigenvectors & Eigenvalues
4.  Sort the eigenvalues (PCs) in descending order based on their significance or strength
5.  Choose the first k eigenvectors

**Step 1: Standardize the data set**

> Standardisation or Z-Score is the process of standardizing every value in a dataset such that the **mean** of all the values is 0 and the Standard Deviation (SD) is 1.

_Why data standardisation is required?_

The very first step in PCA is to standardize the data and ensure all features are normally distributed. Otherwise, it cannot be comparable with the other feature.

Let's say, we compare cooking oil of two different brands' price ranges quantifying one in liters and the other in kilograms. It is incomparable due to different range metrics (Liters & Kilograms). Hence, standardizing those two features helps track metrics properly.

For another instance, you and your friend study in two different colleges.

-   You got a grade of 94 in an exam with a **mean** of 85 and a **Standard Deviation (SD)** of 7
-   Your friend got a grade of 610 with a **mean** of 600 and a **Standard Deviation (SD)** of 100.

It cannot be compared directly! Standardized values will help to compare who is doing better by analyzing two grades.

-   Your grade standardisation : **(94 — 85) / 7 = 1.29**
-   Your friend’s grade standardisation : **(610 — 600) / 100 = 0.10**

With Standardize data, it tells us you are doing better than your friend in the class.

Load our dataset using Pandas

The next step is finding the Covariance.

**Step 2: Covariance Calculations**

_Why do we bother about finding Covariance?_

Covariance calculations are used to find relationships between dimensions in the high dimensional (above 3 dimensions) dataset. It is difficult to visualize high-dimensional datasets!

A **positive covariance** indicates both the dimensions increase or decrease together. For example, _when the number of hours of study increases, the mark in the subject increases._

![](https://miro.medium.com/v2/resize:fit:323/1*wA6J2BTcfwiaXL14Mmth6g.png)

Image: Positive & Negative Covariance

A **negative covariance** indicates while one increases the other decreases. _When a price is low, the demand increases._

**With a zero covariance**, the two dimensions are independent of each other. For example, the _height of students versus the marks of students._

Covariance is always measured between two dimensions. If we have a 3-dimensional data set (_x_, _y_, _z_), then measure the covariance between

-   _x_ and _y_ dimensions
-   _x_ and _z_ dimensions
-   _y_ and _z_ dimensions

The formula for Covariance:

![](https://miro.medium.com/v2/resize:fit:334/1*XIfScFu2_kO7oUGtB4BvUg.png)

Let’s use a simple example:

![](https://miro.medium.com/v2/resize:fit:548/1*lfzmRzlTUhaxbLegYdajVg.png)

After calculating covariance,

\= 4.6 / 3 = 1.53. Hence, it is Positive Covariance which means,

-   When economic growth increases then the Nifty 50 index increases
-   When economic growth decreases then the Nifty 50 index decreases.

3-Dimensional data set using the dimensions _x_, _y,_ and _z._ The covariance matrix has 3 rows and 3 columns, and the values are:

![](https://miro.medium.com/v2/resize:fit:400/1*aTjPWn8-IXYTUl_snX5wuw.png)

From our dataset, PCA normalizes the data and creates a covariance matrix. It is useful to visually see the linear relationships between features.

![](https://miro.medium.com/v2/resize:fit:700/1*BMRUKlp7gqvMxOvq_QHEbw.png)

Image: Visual Covariance matrix

In the above heatmap covariance matrix

-   The diagonal values have a high variance
-   In the non-diagonal values, 91% of covariance between the features _BounceRates_ & _ExitRates (Circled in Red)._

**Step 3: Identifying Principal Components by calculating Eigenvectors and Eigenvalues**

**Why we use Eigenvectors & Eigenvalues?**

To understand a thing, we tend to breakdown into smaller components! When we break down things into their important components, we get a good understanding of a thing. Eigenvectors and Eigenvalues from covariance matrices determine a set of important variables with their dimension and scale.

**For example**, If we need to understand **how a car operates**, we need to breakdown into its important components such as the **engine**, **tires**, **gearbox**, **brake,** etc., and the non-important components in the car like a **sunroof**, **color**, **stereo systems,** etc.,

**When to use Eigenvectors & Eigenvalues?**

In a multi-dimensional dataset with many features, you may want to use eigenvectors to reduce the dimensionality of the data or cluster the features (group) based on similarity or give weights to each feature.

In our original dataset example, **_pca.factors_** returns an array of all principal components.

Correlation matrix

![](https://miro.medium.com/v2/resize:fit:700/1*qOSCREornL_PWC3Mf7zKIw.png)

Image: Correlation matrix for principal components

In the above correlation matrix, Component _comp\_00_ has the highest correlation values.

-   _comp\_00, BounceRates & ExitRates_ — Have high correlation in the above heat map
-   _comp\_01, ProductRelates, ProductRelated\_Duration_ — Have next high correlation.

In PCA, eigenvalues and eigenvectors of the features from the covariance matrix are processed and determined as top K eigenvectors based on the corresponding eigenvalues.

**Step 4: Sort the eigenvalues (PCs) in descending order based on their significance or strength**

![](https://miro.medium.com/v2/resize:fit:143/1*H_bO8d3Yr-DQgUwEQmJx1Q.png)

Image: Online shoppers Dataset Eigenvalues Descending Order Values List

**Step 5: Choose the first k eigenvectors**

In the above car example, top K eigenvectors (important features) are based on weights to determine important features in descending order.

-   **Engine**
-   **Tires**
-   **Gearbox**
-   **Brake**
-   Sunroof
-   Colour
-   Stereo systems

The above list, highlighted in bold are top K eigenvectors! Logically, we know what are all the important principal components to be chosen from the car. In complex eigenvectors or features, it is difficult to identify.

> Mathematically, How to determine the important features?

There are three types that can be used to pick up the top K eigenvectors or features called [Factor Analysis.](https://en.wikipedia.org/wiki/Factor_analysis)

-   Kaiser Criterion
-   Explained Variance
-   Scree Plot

In this shopper online example, we will use a scree plot to visually pick components and use the elbow method to define the point.

Image: Scree plot method to identify Top K values

![](https://miro.medium.com/v2/resize:fit:559/1*ZRFpPxb_FMxJhqK32O5xIQ.png)

Image: Elbow technique in the scree plot

In the shopper's dataset, using the elbow technique to choose #10 components.

![](https://miro.medium.com/v2/resize:fit:162/1*ykvH12GgUF5SxEzHkEUymQ.png)

Image: Top K Eigenvalues — ( 0 to 10)

**Summary**

-   Principal Components Analysis (PCA), is the technique used to do exploratory analysis, image compression, dimensionality reductions
-   In this blog, we have used the Online shoppers purchasing intention dataset
-   Identifying important features using PCA follows 5 steps.

1.  Standardize the data set
2.  Find Covariance
3.  Identify Principal Components (PCs) by calculating Eigenvectors & Eigenvalues
4.  Sort the eigenvalues (PCs) in descending order based on their significance or strength
5.  Choose the first k eigenvalues

-   To identify top K eigenvalues, we can use **Kaiser Criterion** or **Explained Variance**, or **Scree Plot** to choose top K values
-   In our example, we have used PCA for exploratory analysis to identify important features and decide why last year's revenue of online shoppers went so low.
-   Based on the top K principle components analysis, we got important correlations from the top 2 components

1.  Bounce Rates & Exit Rates - Component 00
2.  Product Related & Product Related Duration — Component 01

-   The report clearly says that the root cause for the last year’s low revenue is more customer page bounce rates and exit page rates being very high. And also, customers spending time on the product page is high.
-   To increase the revenue, the eCommerce site owner must improve the product information sites and provide product offers/discount helps to give purchase intent to customers.

Please refer to the [GitHub link](https://github.com/RaghavPrabhu/Deep-Learning/tree/master/pca_exploratory_analysis) for the source code notebook and its dataset.
