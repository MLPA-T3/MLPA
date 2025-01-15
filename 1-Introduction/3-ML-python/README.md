---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.6.5
  nbformat: 4
  nbformat_minor: 1
---

::: {.cell .markdown _cell_guid="dcc6c619-c02a-4bea-b322-4f32779ae672" _uuid="819c6b57207e720308ea29491d85a6e8af8c253e"}
# DATA SCIENTIST

**In this tutorial, I only explain you what you need to be a data
scientist neither more nor less.**

Data scientist need to have these skills:

1.  Basic Tools: Like python, R or SQL. You do not need to know
    everything. What you only need is to learn how to use **python**
2.  Basic Statistics: Like mean, median or standart deviation. If you
    know basic statistics, you can use **python** easily.
3.  Data Munging: Working with messy and difficult data. Like a
    inconsistent date and string formatting. As you guess, **python**
    helps us.
4.  Data Visualization: Title is actually explanatory. We will visualize
    the data with **python** like matplot and seaborn libraries.
5.  Machine Learning: You do not need to understand math behind the
    machine learning technique. You only need is understanding basics of
    machine learning and learning how to implement it while using
    **python**.

### As a summary we will learn python to be data scientist !!! {#as-a-summary-we-will-learn-python-to-be-data-scientist-}

## For parts 1, 2, 3, 4, 5 and 6, look at DATA SCIENCE TUTORIAL for BEGINNERS

<https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/>

## In this tutorial, I am not going to learn machine learning to you, I am going to explain how to learn something by yourself. {#in-this-tutorial-i-am-not-going-to-learn-machine-learning-to-you-i-am-going-to-explain-how-to-learn-something-by-yourself}

# *Confucius: Give a man a fish, and you feed him for a day. Teach a man to fish, and you feed him for a lifetime* {#confucius-give-a-man-a-fish-and-you-feed-him-for-a-day-teach-a-man-to-fish-and-you-feed-him-for-a-lifetime}

**Content:**

1.  Introduction to Python:
    1.  Matplotlib
    2.  Dictionaries
    3.  Pandas
    4.  Logic, control flow and filtering
    5.  Loop data structures
2.  Python Data Science Toolbox:
    1.  User defined function
    2.  Scope
    3.  Nested function
    4.  Default and flexible arguments
    5.  Lambda function
    6.  Anonymous function
    7.  Iterators
    8.  List comprehension
3.  Cleaning Data
    1.  Diagnose data for cleaning
    2.  Explotary data analysis
    3.  Visual exploratory data analysis
    4.  Tidy data
    5.  Pivoting data
    6.  Concatenating data
    7.  Data types
    8.  Missing data and testing with assert
4.  Pandas Foundation
    1.  Review of pandas
    2.  Building data frames from scratch
    3.  Visual exploratory data analysis
    4.  Statistical explatory data analysis
    5.  Indexing pandas time series
    6.  Resampling pandas time series
5.  Manipulating Data Frames with Pandas
    1.  Indexing data frames
    2.  Slicing data frames
    3.  Filtering data frames
    4.  Transforming data frames
    5.  Index objects and labeled data
    6.  Hierarchical indexing
    7.  Pivoting data frames
    8.  Stacking and unstacking data frames
    9.  Melting data frames
    10. Categoricals and groupby
6.  Data Visualization
    1.  Seaborn: <https://www.kaggle.com/kanncaa1/seaborn-for-beginners>
    2.  Bokeh:
        <https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-1>
    3.  Bokeh:
        <https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-2>
7.  Statistical Thinking
    1.  <https://www.kaggle.com/kanncaa1/basic-statistic-tutorial-for-beginners>
8.  [Machine Learning](#1)
    1.  [Supervised Learning](#2)
        1.  [EDA(Exploratory Data Analysis)](#3)
        2.  [K-Nearest Neighbors (KNN)](#4)
        3.  [Regression](#5)
        4.  [Cross Validation (CV)](#6)
        5.  [ROC Curve](#7)
        6.  [Hyperparameter Tuning](#8)
        7.  [Pre-procesing Data](#9)
    2.  [Unsupervised Learning](#10)
        1.  [Kmeans Clustering](#11)
        2.  [Evaluation of Clustering](#12)
        3.  [Standardization](#13)
        4.  [Hierachy](#14)
        5.  [T - Distributed Stochastic Neighbor Embedding (T -
            SNE)](#15)
        6.  [Principle Component Analysis (PCA)](#16)
9.  Deep Learning
    1.  <https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners>
10. Time Series Prediction
    1.  <https://www.kaggle.com/kanncaa1/time-series-prediction-tutorial-with-eda>
11. Deep Learning with Pytorch
    1.  Artificial Neural Network:
        <https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers>
    2.  Convolutional Neural Network:
        <https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers>
    3.  Recurrent Neural Network:
        <https://www.kaggle.com/kanncaa1/recurrent-neural-network-with-pytorch>
:::

::: {.cell .code _cell_guid="5ee3a7aa-eca4-411b-9f84-d14c09e13730" _uuid="2b90d6250c8f9c2c302c849bffa132bd3483e893" collapsed="true" trusted="true"}
``` python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
```
:::

::: {.cell .code _cell_guid="32af03f6-41be-41ec-9023-8cd519040984" _uuid="a9c5426e9e5cef81c1e1639ebe57e9b45dfd2c43" collapsed="true" trusted="false"}
``` python
# read csv (comma separated value) into data
data = pd.read_csv('../input/column_2C_weka.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')
```
:::

::: {.cell .markdown _cell_guid="89a724e2-426d-427b-9107-06835010cf59" _uuid="f4fa42e9a6cf069d54459be42a1726ab03c2f1e5"}
`<a id="1">`{=html}`</a>`{=html} `<br>`{=html}

# 8. MACHINE LEARNING (ML) {#8-machine-learning-ml}

In python there are some ML libraries like sklearn, keras or tensorflow.
We will use sklearn.
:::

::: {.cell .markdown _cell_guid="10f3b719-c44b-451a-b464-0adf4e1e1522" _uuid="f6a2b205e1e3fc647bc2ee88d702b8572bf1cc75"}
`<a id="2">`{=html}`</a>`{=html} `<br>`{=html}

## A. SUPERVISED LEARNING {#a-supervised-learning}

-   Supervised learning: It uses data that has labels. Example, there
    are orthopedic patients data that have labels *normal* and
    *abnormal*.
    -   There are features(predictor variable) and target variable.
        Features are like *pelvic radius* or *sacral slope*(If you have
        no idea what these are like me, you can look images in google
        like what I did :) )Target variables are labels *normal* and
        *abnormal*
    -   Aim is that as given features(input) predict whether target
        variable(output) is *normal* or *abnormal*
    -   Classification: target variable consists of categories like
        normal or abnormal
    -   Regression: target variable is continious like stock market
    -   If these explanations are not enough for you, just google them.
        However, be careful about terminology: features = predictor
        variable = independent variable = columns = inputs. target
        variable = responce variable = class = dependent variable =
        output = result
:::

::: {.cell .markdown _cell_guid="65e897a1-8259-44c5-9cb7-e5e653f9032d" _uuid="a0e671bf2ef8dbe81da2705ad70b69401bb7af16"}
`<a id="3">`{=html}`</a>`{=html} `<br>`{=html}

### EXPLORATORY DATA ANALYSIS (EDA)

-   In order to make something in data, as you know you need to explore
    data. Detailed exploratory data analysis is in my Data Science
    Tutorial for Beginners
-   I always start with *head()* to see features that are
    *pelvic_incidence, pelvic_tilt numeric, lumbar_lordosis_angle,
    sacral_slope, pelvic_radius* and *degree_spondylolisthesis* and
    target variable that is *class*
-   head(): default value of it shows first 5 rows(samples). If you want
    to see for example 100 rows just write head(100)
:::

::: {.cell .code _cell_guid="c1ecd622-67cc-485f-bfa7-8c682d30a5eb" _uuid="9a5993f4962882e1156f2062b7abf706a0739d51" collapsed="true" trusted="false"}
``` python
# to see features and target variable
data.head()
```
:::

::: {.cell .code _cell_guid="1631690c-bb9d-4460-a7d9-a335aa914b4f" _uuid="b7b9addc824de1a35b67d96d3092ffcb10869947" collapsed="true" trusted="false"}
``` python
# Well know question is is there any NaN value and length of this data so lets look at info
data.info()
```
:::

::: {.cell .markdown _cell_guid="a7dd2a6f-a81d-4dce-9d74-fd0148c446ae" _uuid="96f97c33305956eb76d8a2043fd71aff05e38548"}
As you can see:

-   length: 310 (range index)
-   Features are float
-   Target variables are object that is like string
-   Okey we have some ideas about data but lets look go inside data
    deeper
    -   describe(): I explain it in previous tutorial so there is a Quiz
        :)
        -   Why we need to see statistics like mean, std, max or min? I
            hate from quizzes :) so answer: In order to visualize data,
            values should be closer each other. As you can see values
            looks like closer. At least there is no incompatible values
            like mean of one feature is 0.1 and other is 1000. Also
            there are another reasons that I will mention next parts.
:::

::: {.cell .code _cell_guid="137897ca-b519-4ac3-afdd-6c4136447e39" _uuid="69d132068cce6a915aac7678b4e9fbcf3e365643" collapsed="true" trusted="false"}
``` python
data.describe()
```
:::

::: {.cell .markdown _cell_guid="3776dd3d-d0aa-419e-b788-e75454e94b86" _uuid="c8961b1f3a3a73547a0b7d27955f9844f6ad43eb"}
pd.plotting.scatter_matrix:

-   green: *normal* and red: *abnormal*
-   c: color
-   figsize: figure size
-   diagonal: histohram of each features
-   alpha: opacity
-   s: size of marker
-   marker: marker type
:::

::: {.cell .code _cell_guid="fb106765-bb47-452b-8d6e-3578b479873c" _uuid="5dc0763dde8b2638a5289f0b4496f384f85aca85" collapsed="true" trusted="false"}
``` python
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()
```
:::

::: {.cell .markdown _cell_guid="53fc7ab6-de8b-4b8f-9e4a-38c5db72eea0" _uuid="d77cef37b8c5c4f07d3f4aa94cc4ad1ccbd7caca"}
Okay, as you understand in scatter matrix there are relations between
each feature but how many *normal(green)* and *abnormal(red)* classes
are there.

-   Searborn library has *countplot()* that counts number of classes
-   Also you can print it with *value_counts()* method

`<br>`{=html} This data looks like balanced. Actually there is no
definiton or numeric value of balanced data but this data is balanced
enough for us. `<br>`{=html} Now lets learn first classification method
KNN
:::

::: {.cell .code _cell_guid="36243fa5-1fa6-4f8b-bc16-43449b0dc898" _uuid="e1bb9fd338e6900888e2e4717f54be46cee848a2" collapsed="true" trusted="false"}
``` python
sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
```
:::

::: {.cell .markdown _cell_guid="24a5d90f-3e7d-4733-a6f9-ff4f51145155" _uuid="9263c479815bb729dad40bf01b68aa18a3c946ac"}
`<a id="4">`{=html}`</a>`{=html} `<br>`{=html}

### K-NEAREST NEIGHBORS (KNN)

-   KNN: Look at the K closest labeled data points
-   Classification method.
-   First we need to train our data. Train = fit
-   fit(): fits the data, train the data.
-   predict(): predicts the data `<br>`{=html} If you do not understand
    what is KNN, look at youtube there are videos like 4-5 minutes. You
    can understand better with it. `<br>`{=html} Lets learn how to
    implement it with sklearn
-   x: features
-   y: target variables(normal, abnormal)
-   n_neighbors: K. In this example it is 3. it means that Look at the 3
    closest labeled data points
:::

::: {.cell .code _cell_guid="c717491d-2bd5-4dc7-ac13-b9f581b1cddd" _uuid="854f0a3898a928640b9714fcd584e48c9b377f9f" collapsed="true" trusted="false"}
``` python
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))
```
:::

::: {.cell .markdown _cell_guid="b5d85f4e-ab30-4c49-b2bc-6265b6baea9d" _uuid="6d9c3eacd279ddf7c0ef33b9e1814cd549d0feaa"}
-   Well, we fit the data and predict it with KNN.
-   So, do we predict correct or what is our accuracy or the accuracy is
    best metric to evaluate our result? Lets give answer of this
    questions `<br>`{=html} Measuring model performance:
-   Accuracy which is fraction of correct predictions is commonly used
    metric. We will use it know but there is another problem

`<br>`{=html}As you see I train data with x (features) and again predict
the x(features). Yes you are reading right but yes you are right again
it is absurd :)

`<br>`{=html}Therefore we need to split our data train and test sets.

-   train: use train set by fitting
-   test: make prediction on test set.
-   With train and test sets, fitted data and tested data are completely
    different
-   train_test_split(x,y,test_size = 0.3,random_state = 1)
    -   x: features
    -   y: target variables (normal,abnormal)
    -   test_size: percentage of test size. Example test_size = 0.3,
        test size = 30% and train size = 70%
    -   random_state: sets a seed. If this seed is same number,
        train_test_split() produce exact same split at each time
-   fit(x_train,y_train): fit on train sets
-   score(x_test,y_test)): predict and give accuracy on test sets
:::

::: {.cell .code _cell_guid="79865658-c89f-43c8-a1a9-75acc4feab6a" _uuid="4702429fdfa62650937b09fde5a8fd3136da8c55" collapsed="true" trusted="false"}
``` python
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
```
:::

::: {.cell .markdown _cell_guid="a5665258-3f7f-435a-a634-49eb0c0d66e0" _uuid="544f51ef05efe0b3ae4b02da806778bcfa715f35"}
Accuracy is 86% so is it good ? I do not know actually, we will see at
the end of tutorial. `<br>`{=html} Now the question is why we choose K =
3 or what value we need to choose K. The answer is in model complexity

`<br>`{=html} Model complexity:

-   K has general name. It is called a hyperparameter. For now just know
    K is hyperparameter and we need to choose it that gives best
    performace.
-   Literature says if k is small, model is complex model can lead to
    overfit. It means that model memorizes the train sets and cannot
    predict test set with good accuracy.
-   If k is big, model that is less complex model can lead to underfit.
-   At below, I range K value from 1 to 25(exclude) and find accuracy
    for each K value. As you can see in plot, when K is 1 it memozize
    train sets and cannot give good accuracy on test set (overfit). Also
    if K is 18, model is lead to underfit. Again accuracy is not enough.
    However look at when K is 18(best performance), accuracy has highest
    value almost 88%.
:::

::: {.cell .code _cell_guid="db2c7062-ce1b-4e8b-9b2f-0ee0cd679a91" _uuid="18d8739373085a9964071f38b8f2adcb64f25491" collapsed="true" trusted="false"}
``` python
# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
```
:::

::: {.cell .markdown _cell_guid="b598ee81-e535-49c0-b53c-b13b0a5058db" _uuid="96423b4f710966c8071647874623c139c1c79bb7"}
### Up to this point what you learn:

-   Supervised learning
-   Exploratory data analysis
-   KNN
    -   How to split data
    -   How to fit, predict data
    -   How to measure medel performance (accuracy)
    -   How to choose hyperparameter (K)

**`<br>`{=html} What happens if I chance the title KNN and make it some
other classification technique like Random Forest?**

-   The answer is **nothing**. What you need to is just watch a video
    about what is random forest in youtube and implement what you learn
    in KNN. Because the idea and even most of the codes (only
    KNeighborsClassifier need to be RandomForestClassifier ) are same.
    You need to split, fit, predict your data and measue performance and
    choose hyperparameter of random forest(like max_depth).
:::

::: {.cell .markdown _cell_guid="f9d427a9-faa5-46cf-9e3c-2c8cea2571ad" _uuid="d075fd2a7c05e5414e33b7b1314d81a6b945e7b3"}
`<a id="5">`{=html}`</a>`{=html} `<br>`{=html}

### REGRESSION

-   Supervised learning
-   We will learn linear and logistic regressions
-   This orthopedic patients data is not proper for regression so I only
    use two features that are *sacral_slope* and *pelvic_incidence* of
    abnormal
    -   I consider feature is pelvic_incidence and target is
        sacral_slope
    -   Lets look at scatter plot so as to understand it better
    -   reshape(-1,1): If you do not use it shape of x or y becaomes
        (210,) and we cannot use it in sklearn, so we use shape(-1,1)
        and shape of x or y be (210, 1).
:::

::: {.cell .code _cell_guid="6b072c42-059f-4e45-9cfa-8ed39b274f72" _uuid="d2655c140423b1228c42d2e8dfe54344ba43dcb0" collapsed="true" trusted="false"}
``` python
# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] =='Abnormal']
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="874ac5e0-5bc0-4429-b6c5-707690b5dd77" _uuid="c84719d363ff736e96a0a575dfd0381bcbc549fb"}
Now we have our data to make regression. In regression problems target
value is continuously varying variable such as price of house or
sacral_slope. Lets fit line into this points.

`<br>`{=html} Linear regression

-   y = ax + b where y = target, x = feature and a = parameter of model
-   We choose parameter of model(a) according to minimum error function
    that is lost function
-   In linear regression we use Ordinary Least Square (OLS) as lost
    function.
-   OLS: sum all residuals but some positive and negative residuals can
    cancel each other so we sum of square of residuals. It is called OLS
-   Score: Score uses R\^2 method that is ((y_pred - y_mean)\^2
    )/(y_actual - y_mean)\^2
:::

::: {.cell .code _cell_guid="fb7991f3-5869-4df0-bf6c-30f61e8215c6" _uuid="7cdc74efa8c46dab5f14f6cc2779928c11a4fa62" collapsed="true" trusted="false"}
``` python
# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# Predict space
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
# Fit
reg.fit(x,y)
# Predict
predicted = reg.predict(predict_space)
# R^2 
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="242dd057-bb8a-463a-b04c-1072d5de1e0a" _uuid="3cf254d2feef8f9f886b0e1de12ec2b155083d2b"}
`<a id="6">`{=html}`</a>`{=html} `<br>`{=html}

### CROSS VALIDATION

As you know in KNN method we use train test split with random_state that
split exactly same at each time. However, if we do not use random_state,
data is split differently at each time and according to split accuracy
will be different. Therefore, we can conclude that model performance is
dependent on train_test_split. For example you split, fit and predict
data 5 times and accuracies are 0.89, 0.9, 0.91, 0.92 and 0.93,
respectively. Which accuracy do you use? Do you know what accuracy will
be at 6th times split, train and predict. The answer is I do not know
but if I use cross validation I can find acceptable accuracy.
`<br>`{=html} Cross Validation (CV)

-   K folds = K fold CV.
-   Look at this image it defines better than me :)
-   When K is increase, computationally cost is increase
-   cross_val_score(reg,x,y,cv=5): use reg(linear regression) with x and
    y that we define at above and K is 5. It means 5 times(split,
    train,predict)
:::

::: {.cell .code _cell_guid="bf504792-66f4-411e-bc1d-5b049da959ac" _uuid="1f1a77c5d5e6ca52c0264875362665f228e66078" collapsed="true" trusted="false"}
``` python
# CV
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)
```
:::

::: {.cell .markdown _cell_guid="27ffec30-4951-4479-9d86-32495a80c08d" _uuid="be425f142c0acc1ab3009fafafa3616948f5c8a5"}
### Regularized Regression

As we learn linear regression choose parameters (coefficients) while
minimizing lost function. If linear regression thinks that one of the
feature is important, it gives high coefficient to this feature.
However, this can cause overfitting that is like memorizing in KNN. In
order to avoid overfitting, we use regularization that penalize large
coefficients.

-   Ridge regression: First regularization technique. Also it is called
    L2 regularization.
    -   Ridge regression lost fuction = OLS + alpha \* sum(parameter\^2)
    -   alpha is parameter we need to choose to fit and predict. Picking
        alpha is similar to picking K in KNN. As you understand alpha is
        hyperparameter that we need to choose for best accuracy and
        model complexity. This process is called hyperparameter tuning.
    -   What if alpha is zero? lost function = OLS so that is linear
        rigression :)
    -   If alpha is small that can cause overfitting
    -   If alpha is big that can cause underfitting. But do not ask what
        is small and big. These can be change from problem to problem.
-   Lasso regression: Second regularization technique. Also it is called
    L1 regularization.
    -   Lasso regression lost fuction = OLS + alpha \*
        sum(absolute_value(parameter))
    -   It can be used to select important features od the data. Because
        features whose values are not shrinked to zero, is chosen by
        lasso regression
    -   In order to choose feature, I add new features in our regression
        data

`<br>`{=html} Linear vs Ridge vs Lasso First impression: Linear Feature
Selection: 1.Lasso 2.Ridge Regression model: 1.Ridge 2.Lasso 3.Linear
:::

::: {.cell .code _cell_guid="bdc1a379-07a4-4b61-8ac1-d7007ae33783" _uuid="85fa872e3988295a8fbe752bf96319518ca3595b" collapsed="true" trusted="false"}
``` python
# Ridge
from sklearn.linear_model import Ridge
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(x_train,y_train)
ridge_predict = ridge.predict(x_test)
print('Ridge score: ',ridge.score(x_test,y_test))
```
:::

::: {.cell .code _cell_guid="de031d9f-b2f8-4fb1-a305-36cbc4dc970f" _uuid="57f91f4b4e267bd3eb22adfcdf719778c1901c92" collapsed="true" trusted="false"}
``` python
# Lasso
from sklearn.linear_model import Lasso
x = np.array(data1.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']])
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)
lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(x_train,y_train)
ridge_predict = lasso.predict(x_test)
print('Lasso score: ',lasso.score(x_test,y_test))
print('Lasso coefficients: ',lasso.coef_)
```
:::

::: {.cell .markdown _cell_guid="84f9d88d-28e9-4b26-8824-ae32de7e143c" _uuid="d70e2366ccc8797c92d2edacb4ab3b59fad4506d"}
As you can see *pelvic_incidence* and *pelvic_tilt numeric* are
important features but others are not important

`<br>`{=html} Now lets discuss accuracy. Is it enough for measurement of
model selection. For example, there is a data that includes 95% normal
and 5% abnormal samples and our model uses accuracy for measurement
metric. Then our model predict 100% normal for all samples and accuracy
is 95% but it classify all abnormal samples wrong. Therefore we need to
use confusion matrix as a model measurement matris in imbalance data.
`<br>`{=html} While using confusion matrix lets use Random forest
classifier to diversify classification methods.

-   tp = true positive(20), fp = false positive(7), fn = false
    negative(8), tn = true negative(58)
-   tp = Prediction is positive(normal) and actual is positive(normal).
-   fp = Prediction is positive(normal) and actual is
    negative(abnormal).
-   fn = Prediction is negative(abnormal) and actual is
    positive(normal).
-   tn = Prediction is negative(abnormal) and actual is
    negative(abnormal)
-   precision = tp / (tp+fp)
-   recall = tp / (tp+fn)
-   f1 = 2 \* precision \* recall / ( precision + recall)
:::

::: {.cell .code _cell_guid="c71a33f5-5784-461b-949d-cb83f23dace6" _uuid="19fb6fb0f651e249835037a4f4b2f0b4a2619a27" collapsed="true" trusted="false"}
``` python
# Confusion matrix with random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))
```
:::

::: {.cell .code _cell_guid="fcba81bb-1257-48d0-afe1-a4416237cc73" _uuid="f2697bbc248102687596713406512b4cb7f24929" collapsed="true" trusted="false"}
``` python
# visualize with seaborn library
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()
```
:::

::: {.cell .markdown _cell_guid="4552b965-e15d-4e05-9e02-224159e8d508" _uuid="dbe175099c0c8151c16ad0c78f1414de8fd9ebdc"}
`<a id="7">`{=html}`</a>`{=html} `<br>`{=html}

### ROC Curve with Logistic Regression

-   logistic regression output is probabilities
-   If probability is higher than 0.5 data is labeled 1(abnormal) else
    0(normal)
-   By default logistic regression threshold is 0.5
-   ROC is receiver operationg characteristic. In this curve x axis is
    false positive rate and y axis is true positive rate
-   If the curve in plot is closer to left-top corner, test is more
    accurate.
-   Roc curve score is auc that is computation area under the curve from
    prediction scores
-   We want auc to closer 1
-   fpr = False Positive Rate
-   tpr = True Positive Rate
-   If you want, I made ROC, Random forest and K fold CV in this
    tutorial.
    <https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv/>
:::

::: {.cell .code _cell_guid="4f4a8f76-9792-485f-83b0-79e2524ab83c" _uuid="c7fce3a219764388088ec5f3d57ab913f5c05f35" collapsed="true" trusted="false"}
``` python
# ROC Curve with logistic regression
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# abnormal = 1 and normal = 0
data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]
x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_prob = logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="ceb21da5-4ced-400c-a017-60f735802b69" _uuid="db2a0e5e83aee71a9585ee113cb01170325c96f1"}
`<a id="8">`{=html}`</a>`{=html} `<br>`{=html}

### HYPERPARAMETER TUNING

As I mention at KNN there are hyperparameters that are need to be tuned

-   For example:
    -   k at KNN
    -   alpha at Ridge and Lasso
    -   Random forest parameters like max_depth
    -   linear regression parameters(coefficients)
-   Hyperparameter tuning:
    -   try all of combinations of different parameters
    -   fit all of them
    -   measure prediction performance
    -   see how well each performs
    -   finally choose best hyperparameters
-   This process is most difficult part of this tutorial. Because we
    will write a lot of for loops to iterate all combinations. Just I am
    kidding sorry for this :) (We actually did it at KNN part)
-   We only need is one line code that is GridSearchCV
    -   grid: K is from 1 to 50(exclude)
    -   GridSearchCV takes knn and grid and makes grid search. It means
        combination of all hyperparameters. Here it is k.
:::

::: {.cell .code _cell_guid="d1a56a4e-a307-4376-8d06-53fc30447ce6" _uuid="20fb485285b4d27da2e3eb89e8e43797db7c457a" collapsed="true" trusted="false"}
``` python
# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x,y)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
```
:::

::: {.cell .markdown _cell_guid="a4401913-229c-4bcc-8dd0-3563b824f6e9" _uuid="bdc9c267b635db5696e73cb6ba36d093c8f127c5"}
Other grid search example with 2 hyperparameter

-   First hyperparameter is C:logistic regression regularization
    parameter
    -   If C is high: overfit
    -   If C is low: underfit
-   Second hyperparameter is penalty(lost function): l1 (Lasso) or
    l2(Ridge) as we learnt at linear regression part.
:::

::: {.cell .code _cell_guid="8988b749-3bf5-448f-830e-edce366348d0" _uuid="eb5961ae83511410ae491366bd12f70bef9f440b" collapsed="true" trusted="false"}
``` python
# grid search cross validation with 2 hyperparameter
# 1. hyperparameter is C:logistic regression regularization parameter
# 2. penalty l1 or l2
# Hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=3)
logreg_cv.fit(x_train,y_train)

# Print the optimal parameters and best score
print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))
```
:::

::: {.cell .markdown _cell_guid="37171c8f-21ad-4628-8279-5cfedb27aafb" _uuid="7cd0bf0ed62c1858f5de74ee4f85c84765f8bc06"}
`<a id="9">`{=html}`</a>`{=html} `<br>`{=html}

### PRE-PROCESSING DATA

-   In real life data can include objects or categorical data in order
    to use them in sklearn we need to encode them into numerical data
-   In data, class is *abnormal* and *normal*. Lets convert them into
    numeric value (actually I did it in logistic regression part with
    different method)
-   2 different feature is created with the name *class_Abnormal* and
    *class_Normal*
-   However we need to drop one of the column because they are
    duplicated
:::

::: {.cell .code _cell_guid="342e6f67-afeb-42b0-9e56-6521fd15e3ad" _uuid="8168cf770134142252e92022307815e31a931a34" collapsed="true" trusted="false"}
``` python
# Load data
data = pd.read_csv('../input/column_2C_weka.csv')
# get_dummies
df = pd.get_dummies(data)
df.head(10)
```
:::

::: {.cell .code _cell_guid="e2541cfc-7d99-45ce-8486-80f7fec79257" _uuid="032c522e3f423822402c7f75fcfe3818894f68c8" collapsed="true" trusted="false"}
``` python
# drop one of the feature
df.drop("class_Normal",axis = 1, inplace = True) 
df.head(10)
# instead of two steps we can make it with one step pd.get_dummies(data,drop_first = True)
```
:::

::: {.cell .markdown _cell_guid="f55c171e-36d4-4873-a0f6-e8ee411d0b64" _uuid="5465db779fed984dd89ad591889d1dadb88fc056"}
Other preprocessing step is centering, scaling or normalizing

-   If you listen my advice and watch KNN in youtube, you have noticed
    that KNN uses form of distance for classificaiton like some oher
    methods. Therefore, we need to scale data. For this reason, we use
    -   standardization: ( x - x.mean) / x.variance or x - x.min /
        x.range
-   pipeline: The purpose of the pipeline is to assemble several steps
    like svm(classifier) and standardization(pre-processing)
-   How we create parameters name: for example SVM\_ \_C :
    stepName\_\_parameterName
-   Then grid search to find best parameters
:::

::: {.cell .code _cell_guid="5d4aef88-f75e-4485-b481-bdeabbc0e4c7" _uuid="4690eac9c778ba7230fcee5a782a46396aa8c4ef" collapsed="true" trusted="false"}
``` python
# SVM, pre-process and pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
```
:::

::: {.cell .markdown _cell_guid="b343630f-75c8-4362-8cb0-858e71a2f4b0" _uuid="3a8fede9eadd9bd0438f7cb50c0155491d3f776b" collapsed="true"}
`<a id="10">`{=html}`</a>`{=html} `<br>`{=html}

## UNSUPERVISED LEARNING

-   Unsupervised learning: It uses data that has unlabeled and uncover
    hidden patterns from unlabeled data. Example, there are orthopedic
    patients data that do not have labels. You do not know which
    orthopedic patient is normal or abnormal.
-   As you know orthopedic patients data is labeled (supervised) data.
    It has target variables. In order to work on unsupervised learning,
    lets drop target variables and to visualize just consider
    *pelvic_radius* and *degree_spondylolisthesis*
:::

::: {.cell .markdown _cell_guid="e492f7d0-5d05-4717-a149-e046d9435607" _uuid="f4927a23d072a9416b57f0bd611d71411407a907"}
`<a id="11">`{=html}`</a>`{=html} `<br>`{=html}

### KMEANS

-   Lets try our first unsupervised method that is KMeans Cluster
-   KMeans Cluster: The algorithm works iteratively to assign each data
    point to one of K groups based on the features that are provided.
    Data points are clustered based on feature similarity
-   KMeans(n_clusters = 2): n_clusters = 2 means that create 2 cluster
:::

::: {.cell .code _cell_guid="fa3dce86-b3cd-4a8f-b689-e5bc5f2fe551" _uuid="53175af6bce4acd6f743250c240033db04921ad4" collapsed="true" trusted="false"}
``` python
# As you can see there is no labels in data
data = pd.read_csv('../input/column_2C_weka.csv')
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
```
:::

::: {.cell .code _cell_guid="63a3245d-0056-46a0-ba65-a1f61d84441f" _uuid="9550966d8637f2d5c61633d597cd3746fb294e7d" collapsed="true" trusted="false"}
``` python
# KMeans Clustering
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="fc405787-c073-4377-a4d8-0dcb83371888" _uuid="bb760dd8801ea5d0b7d98d1e22f784a9c999b900"}
`<a id="12">`{=html}`</a>`{=html} `<br>`{=html}

### EVALUATING OF CLUSTERING

We cluster data in two groups. Okey well is that correct clustering? In
order to evaluate clustering we will use cross tabulation table.

-   There are two clusters that are *0* and *1*
-   First class *0* includes 138 abnormal and 100 normal patients
-   Second class *1* includes 72 abnormal and 0 normal patiens \*The
    majority of two clusters are abnormal patients.
:::

::: {.cell .code _cell_guid="df624885-2e0e-474f-82cb-8a498049d8cc" _uuid="063968754033b4510dcb6f90245a1a9688d4c201" collapsed="true" trusted="false"}
``` python
# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
```
:::

::: {.cell .markdown _cell_guid="f5c33609-4573-443a-82c5-5dfe9852c582" _uuid="03df9051e892dc064c706619e796b5c9128c0a68"}
The new question is that we know how many class data includes, but what
if number of class is unknow in data. This is kind of like
hyperparameter in KNN or regressions.

-   inertia: how spread out the clusters are distance from each sample
-   lower inertia means more clusters
-   What is the best number of clusters ? \*There are low inertia and
    not too many cluster trade off so we can choose elbow
:::

::: {.cell .code _cell_guid="710c918b-481b-4f32-94f6-adecd992a97e" _uuid="6c7e94ea35a4f535857aa2551ed646ff8c28c1ac" collapsed="true" trusted="false"}
``` python
# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="6449fdb8-7e0b-4432-9ea3-6d866d899704" _uuid="06031b6c13b5f700bb2e3d53081d990f66c5e11b"}
`<a id="13">`{=html}`</a>`{=html} `<br>`{=html}

### STANDARDIZATION

-   Standardizaton is important for both supervised and unsupervised
    learning
-   Do not forget standardization as pre-processing
-   As we already have visualized data so you got the idea. Now we can
    use all features for clustering.
-   We can use pipeline like supervised learning.
:::

::: {.cell .code _cell_guid="0ae27970-8319-45a7-ba7c-a8dd9d9b427e" _uuid="97a7e3a3fa1722970ff34935470be85f2ec7977b" collapsed="true" trusted="false"}
``` python
data = pd.read_csv('../input/column_2C_weka.csv')
data3 = data.drop('class',axis = 1)
```
:::

::: {.cell .code _cell_guid="65a46d3b-45b5-4bcd-8137-bdd5b6507ab0" _uuid="6883975e95325db10ebed6703c372cbe4da29a48" collapsed="true" trusted="false"}
``` python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
```
:::

::: {.cell .markdown _cell_guid="1a31b9e4-f152-4da1-a132-a6e3a8090401" _uuid="a602f3d3d47f3bcc8b62bf0ed3114a3ca4760a03"}
`<a id="14">`{=html}`</a>`{=html} `<br>`{=html}

### HIERARCHY

-   vertical lines are clusters
-   height on dendogram: distance between merging cluster
-   method= \'single\' : closest points of clusters
:::

::: {.cell .code _cell_guid="f04ec5bb-9f74-4036-9dfd-2649a3519d79" _uuid="a2dc27b105fc443f442b9605237f45ba69d1e1f1" collapsed="true" trusted="false"}
``` python
from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[200:220,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()
```
:::

::: {.cell .markdown _cell_guid="15b4f98e-0c9e-49de-b69f-f583524c19ac" _uuid="4166039422a3ed14b4ac779da29401378dfb4dc2"}
`<a id="15">`{=html}`</a>`{=html} `<br>`{=html}

### T - Distributed Stochastic Neighbor Embedding (T - SNE)

-   learning rate: 50-200 in normal
-   fit_transform: it is both fit and transform. t-sne has only have
    fit_transform
-   Varieties have same position relative to one another
:::

::: {.cell .code _cell_guid="b4ddf817-57ec-45eb-aa03-1ad385cd4f2e" _uuid="311a16a96b2ef3e9ca3f8e91db48f28837577291" collapsed="true" trusted="false"}
``` python
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(data2)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list )
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="00555f09-d381-494e-8bb9-7afb768d912a" _uuid="71d32fd64448612635a015c3a4e41871bddc337d"}
`<a id="16">`{=html}`</a>`{=html} `<br>`{=html}

### PRINCIPLE COMPONENT ANALYSIS (PCA)

-   Fundemental dimension reduction technique
-   first step is decorrelation:
    -   rotates data samples to be aligned with axes
    -   shifts data asmples so they have mean zero
    -   no information lost
    -   fit() : learn how to shift samples
    -   transform(): apply the learned transformation. It can also be
        applies test data
-   Resulting PCA features are not linearly correlated
-   Principle components: directions of variance
:::

::: {.cell .code _cell_guid="93b827bf-616e-436f-a514-86fafbf44512" _uuid="04e33157c4c218d873892090649b169babfd03a8" collapsed="true" trusted="false"}
``` python
# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)
```
:::

::: {.cell .code _cell_guid="fc2fdc3e-a5f6-4d67-979e-b9e9cd2e7198" _uuid="409100f13e8c3e7bea30dab77ad7ea56e2e338e8" collapsed="true" trusted="false"}
``` python
# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()
```
:::

::: {.cell .markdown _cell_guid="3b6f44f2-0add-4f82-a965-859215a9fea9" _uuid="ec0ad8f2341933207e26f9eaced8a45e16f2bacf"}
-   Second step: intrinsic dimension: number of feature needed to
    approximate the data essential idea behind dimension reduction
-   PCA identifies intrinsic dimension when samples have any number of
    features
-   intrinsic dimension = number of PCA feature with significant
    variance
-   In order to choose intrinsic dimension try all of them and find best
    accuracy
-   Also check intuitive way of PCA with this example:
    <https://www.kaggle.com/kanncaa1/tutorial-pca-intuition-and-image-completion>
:::

::: {.cell .code _cell_guid="fa5ec66c-2338-4298-a6f8-364a0703ca1f" _uuid="0f1768fd650fe0b85cdf21140c877471c8baf2bc" collapsed="true" trusted="false"}
``` python
# apply PCA
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()
```
:::

::: {.cell .markdown _cell_guid="41762172-8d1f-47ef-b8ab-79fb3a2b0a19" _uuid="2c1a42c1de45ce53e04761b4b6c05840fddaee95"}
# CONCLUSION

This is the end of DATA SCIENCE tutorial. The first part is here:
`<br>`{=html}
<https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/>
`<br>`{=html}**If you have any question or suggest, I will be happy to
hear it.**
:::
