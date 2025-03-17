
## Introduction to Time Series

A time series is a sequence or series of numerical data points fixed at certain chronological time order. In most cases, a time series is a sequence taken at fixed interval points in time. This allows us to accurately predict or forecast the necessities.

Time series uses line charts to show us seasonal patterns, trends, and relation to external factors. It uses time series values for forecasting and this is called extrapolation.

Time series are used in most of the real-life cases such as weather reports, earthquake prediction, astronomy, mathematical finance, and largely in any field of applied science and engineering. It gives us deeper insights into our field of work and forecasting helps an individual in increasing efficiency of output.

## Time Series Forecasting

_Time series forecasting_ is a method of using a model to predict future values based on previously observed time series values.

Time series is an important part of machine learning. It figures out a seasonal pattern or trend in the observed time-series data and uses it for future predictions or forecasting. Forecasting involves taking models rich in historical data and using them to predict future observations.

One of the most distinctive features of forecasting is that it does not exactly predict the future, it just gives us a calculated estimation of what has already happened to give us an idea of what could happen.

![](https://miro.medium.com/v2/resize:fit:816/1*kptV21R6IRbNoabMkX-57A.png)

Image Courtesy: www.wfmanagement.blogspot.com

Now let’s look at the general forecasting methods used in day to day problems,

_Qualitative forecasting_ is generally used when historical data is unavailable and is considered to be highly objective and judgmental.

_Quantitative forecasting_ is when we have large amounts of data from the past and is considered to be highly efficient as long as there is no strong external factors in play.

The skill of a time series forecasting model is determined by its efficiency at predicting the future. This is often at the cost of being able to explain why a specific prediction was made, confidence intervals, and even better, understanding the underlying factors behind the problem.

Some general examples of forecasting are:

1.  Governments forecast unemployment rates, interest rates, and expected revenues from income taxes for policy purposes.
2.  Day to day weather prediction.
3.  College administrators forecast enrollments to plan for facilities and faculty recruitment.
4.  Industries forecast demand to control inventory levels, hire employees, and provide training.


Let's explore some of the types of time series algorithms and start a notebook to clean and prepare some data. The data you will analyze  is taken from the GEFCom2014 forecasting competition. It consists of 3 years of hourly electricity load and temperature values between 2012 and 2014. Given the historical patterns of electricity load and temperature, you can predict future values of electricity load.

In this example, you'll learn how to forecast one time step ahead, using historical load data only. Before starting, however, it's useful to understand what's going on behind the scenes.

## Some definitions

When encountering the term 'time series' you need to understand its use in several different contexts.

🎓 **Time series**

In mathematics, "a time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time." An example of a time series is the daily closing value of the [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). The use of time series plots and statistical modeling is frequently encountered in signal processing, weather forecasting, earthquake prediction, and other fields where events occur and data points can be plotted over time.

🎓 **Time series analysis**

Time series analysis, is the analysis of the above mentioned time series data. Time series data can take distinct forms, including 'interrupted time series' which detects patterns in a time series' evolution before and after an interrupting event. The type of analysis needed for the time series, depends on the nature of the data. Time series data itself can take the form of series of numbers or characters.

The analysis to be performed, uses a variety of methods, including frequency-domain and time-domain, linear and nonlinear, and more. [Learn more](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) about the many ways to analyze this type of data.

🎓 **Time series forecasting**

Time series forecasting is the use of a model to predict future values based on patterns displayed by previously gathered data as it occurred in the past. While it is possible to use regression models to explore time series data, with time indices as x variables on a plot, such data is best analyzed using special types of models.

Time series data is a list of ordered observations, unlike data that can be analyzed by linear regression.   The most common one is ARIMA, an acronym that stands for "Autoregressive Integrated Moving Average".

[ARIMA models](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relate the present value of a series to past values and past prediction errors." They are most appropriate for analyzing time-domain data, where data is ordered over time.

> There are several types of ARIMA models, which you can learn about [here](https://people.duke.edu/~rnau/411arim.htm) and which you will touch on in the next lesson.

In the next lesson, you will build an ARIMA model using [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), which focuses on one variable that changes its value over time. An example of this type of data is [this dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) that records the monthly C02 concentration at the Mauna Loa Observatory:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identify the variable that changes over time in this dataset

## Time Series data characteristics to consider

When looking at time series data, you might notice that it has [certain characteristics](https://online.stat.psu.edu/stat510/lesson/1/1.1) that you need to take into account and mitigate to better understand its patterns. If you consider time series data as potentially providing a 'signal' that you want to analyze, these characteristics can be thought of as 'noise'. You often will need to reduce this 'noise' by offsetting some of these characteristics using some statistical techniques.

Here are some concepts you should know to be able to work with time series:

🎓 **Trends**

Trends are defined as measurable increases and decreases over time. [Read more](https://machinelearningmastery.com/time-series-trends-in-python). In the context of time series, it's about how to use and, if necessary, remove trends from your time series.

🎓 **[Seasonality](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Seasonality is defined as periodic fluctuations, such as holiday rushes that might affect sales, for example. [Take a look](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) at how different types of plots display seasonality in data.

🎓 **Outliers**

Outliers are far away from the standard data variance.

🎓 **Long-run cycle**

Independent of seasonality, data might display a long-run cycle such as an economic down-turn that lasts longer than a year.

🎓 **Constant variance**

Over time, some data display constant fluctuations, such as energy usage per day and night.

🎓 **Abrupt changes**

The data might display an abrupt change that might need further analysis. The abrupt shuttering of businesses due to COVID, for example, caused changes in data.

✅ Here is a [sample time series plot](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) showing daily in-game currency spent over a few years. Can you identify any of the characteristics listed above in this data?

![In-game currency spend](./images/currency.png)

## Exercise - getting started with power usage data

Let's get started creating a time series model to predict future power usage given past usage.

> The data in this example is taken from the GEFCom2014 forecasting competition. It consists of 3 years of hourly electricity load and temperature values between 2012 and 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli and Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. In the `working` folder of this lesson, open the _notebook.ipynb_ file. Start by adding libraries that will help you load and visualize data

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Note, you are using the files from the included `common` folder which set up your environment and handle downloading the data.

2. Next, examine the data as a dataframe calling `load_data()` and `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    You can see that there are two columns representing date and load:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Now, plot the data calling `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![energy plot](images/energy-plot.png)

4. Now, plot the first week of July 2014, by providing it as input to the `energy` in `[from date]: [to date]` pattern:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![july](images/july-2014.png)

    A beautiful plot! Take a look at these plots and see if you can determine any of the characteristics listed above. What can we surmise by visualizing the data?

In the next lesson, you will create an ARIMA model to create some forecasts.

---

## 🚀Challenge

Make a list of all the industries and areas of inquiry you can think of that would benefit from time series forecasting. Can you think of an application of these techniques in the arts? In Econometrics? Ecology? Retail? Industry? Finance? Where else?

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42/)

## Review & Self Study

Although we won't cover them here, neural networks are sometimes used to enhance classic methods of time series forecasting. Read more about them [in this article](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Assignment

[Visualize some more time series](assignment.md)



## Application of Time Series Forecasting

The usage of time series models is twofold:

-   Obtain an understanding of the underlying forces and structure that produced the data
-   Fit a model and proceed to forecast.

There is almost an endless application of time series forecasting problems.

Below are a few of the examples from a range of industries to make the notions of time series analysis and forecasting more strong.

-   Forecasting the rice yield in tons by the state each year.
-   Forecasting whether an EEG trace in seconds indicates a patient is having a heart attack or not.
-   Forecasting the closing price of stock each day.
-   Forecasting the birth or death rate at all hospitals in a city each year.
-   Forecasting product sales in units sold each day.
-   Forecasting the number of passengers booking flight tickets each day.
-   Forecasting unemployment for a state each quarter
-   Forecasting the size of the tiger population in a state each breeding season.

Now let’s look at an example,

We are going to use the google new year resolution dataset,

**Step 1**: Import Libraries

![](https://miro.medium.com/v2/resize:fit:816/1*rn8mifVsEORA-LMiYaS_kg.jpeg)

Picture 1

**Step 2**: Load Dataset

![](https://miro.medium.com/v2/resize:fit:816/1*qdLDo7qpFYUzpyfMvSkBuQ.jpeg)

Picture 2

**Step 3**: Change month column into the DateTime data type

![](https://miro.medium.com/v2/resize:fit:816/1*Tr5kJLx944uZfzw5qDfBFQ.jpeg)

Picture 3

**Step 4**: Plot and visualize

![](https://miro.medium.com/v2/resize:fit:816/1*SMRp0LH8iiUqG45LmFD4RQ.jpeg)

Picture 4.1

![](https://miro.medium.com/v2/resize:fit:816/1*v9yc20Pqus7lRYlEvsAxyQ.jpeg)

Picture 4.2

**Step 5**: Check for trend

![](https://miro.medium.com/v2/resize:fit:816/1*d0NpwJSPf_sufCvwRntoZw.jpeg)

Picture 5.1

![](https://miro.medium.com/v2/resize:fit:816/1*XPXZnPuNaZTn88fgMOdk_A.jpeg)

Picture 5.2

**Step 6**: Check for seasonality

![](https://miro.medium.com/v2/resize:fit:816/1*S22yB8LBHkrI2FKgQAeyNA.jpeg)

Picture 6.1

![](https://miro.medium.com/v2/resize:fit:816/1*QUoCXGgaml836orFoeIPeA.jpeg)

Picture 6.2

We can see that there is roughly a 20% spike each year, this is seasonality.

## Components of Time Series

Time series analysis provides a ton of techniques to better understand a dataset.

Perhaps the most useful of these is the splitting of time series into 4 parts:

1.  **_Level_**: The base value for the series if it were a straight line.
2.  **_Trend_**: The linear increasing or decreasing behavior of the series over time.
3.  **_Seasonality_**: The repeating patterns or cycles of behavior over time.
4.  **_Noise_**: The variability in the observations that cannot be explained by the model.

All-time series generally have a level, noise, while trend and seasonality are optional.

The main features of many time series are trends and seasonal variation. Another feature of most time series is that observations close together in time tend to be correlated

These components combine in some way to provide the observed time series. For example, they may be added together to form a model such as:

_Y =levels + trends + seasonality + noise_

![](https://miro.medium.com/v2/resize:fit:816/1*eT_u8KERJVTc-SXq_nL3Hw.png)

Image Courtesy: Machine Learning Mastery

These components are the most effective way to make predictions about future values, but may not always work. That depends on the amount of data we have about the past.

## Analyzing Trend

Checking out data for repeated behavior in its graphical representation is known as a Trend analysis. As long as the trend is continuously increasing or decreasing that part of data analysis is generally not very difficult. If the time series data contains some kind of considerable error, then the first step in the process of trend identification is smoothing.

**_Smoothing_**. Smoothing always involves some form of local averaging of data such that the components of individual observations cancel each other out. The most widely used technique is moving average smoothing which replaces each element of the series with a simple or weighted average of surrounding elements. Medians are mostly used instead of means. The main advantage of median as compared to moving average smoothing is that its results are less biased by outliers within the smoothing window. The main disadvantage of median smoothing is that in the absence of clear outliers it may produce more disturbed curves than moving average.

In the other less common cases, when the measurement error is quiet large, the distance weighted least squares smoothing or negative exponentially weighted smoothing techniques might be used. These methods generally tend to ignore outliers and give a smooth fitting curve.

**_Fitting a function_**. If there is a clear monotonous nonlinear component, the data first need to be transformed to remove the nonlinearity. Usually, log, exponential, or polynomial function is used to achieve this.

Now let’s take an example to understand this more clearly,

![](https://miro.medium.com/v2/resize:fit:816/1*n4JGp81Mcah5ZjC5-4H_nw.jpeg)

Picture 7.1

![](https://miro.medium.com/v2/resize:fit:816/1*HFoIJWKjzu6O7XF1h0cFbg.jpeg)

Picture 7.2

From the above diagram, we can easily interpret that there is an _upward_ trend for ‘Gym’ every year!

## Analyzing Seasonality

Seasonality is the repetition of data at a certain period of time interval. For example, every year we notice that people tend to go on vacation during the December — January time, this is seasonality. It is one other most important characteristics of time series analysis. It is generally measured by autocorrelation after subtracting the trend from the data.

Lets look at another example from our dataset,

![](https://miro.medium.com/v2/resize:fit:816/1*2IQDm1KSQ7CSJxRoTg6psw.jpeg)

Picture 8.1

![](https://miro.medium.com/v2/resize:fit:816/1*Bw84qKtIEB-2X0rrUdDm2A.jpeg)

Picture 8.2

From the above graph, it is clear that there is a spike at the starting of every year. Which means every year January people tend to take ‘Diet’ as their resolution rather than any other month. This is a perfect example of seasonality.

## AR, MA, and ARIMA

## Autoregression Model (AR)

AR is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step. A regression model like linear regression takes the form of:

_yhat = b0 + (b1 \* X1)_

This technique can be used on time series where input variables are taken as observations at previous time steps, called lag variables. This would look like:

_Xt+1 = b0 + (b1 \* Xt) + (b2 \* Xt-1)_

Since the regression model uses data from the same input variable at previous time steps, it is referred to as autoregression.

## Moving Average Model (MA)

The residual errors from forecasts in a time series provide another source of information that can be modeled. The Residual errors form a time series. An autoregression model of this structure can be used to foresee the forecast error, which in turn can be used to correct forecasts.

Structure in the residual error may consist of trend, bias & seasonality which can be modeled directly. One can create a model of the residual error time series and predict the expected error of the model. The predicted error can then be subtracted from the model prediction & in turn provide an additional lift in performance.

An autoregression of the residual error is the Moving Average Model.

## Autoregressive Integrated Moving Average (ARIMA)

-   Autoregressive integrated moving average or ARIMA is a very important part of statistics, econometrics, and in particular time series analysis.
-   ARIMA is a forecasting technique that gives us future values entirely based on its inertia.
-   Autoregressive Integrated Moving Average (ARIMA) models include a clear cut statistical model for the asymmetrical component of a time series that allows for non-zero autocorrelations in the irregular component
-   ARIMA models are defined for stationary time series. Therefore, if you start with a non-stationary time series, you will first need to ‘difference’ the time series until you attain stationary time series.

An ARIMA model can be created using the statsmodels library as follows:

1.  Define the model by using **ARIMA()** and passing in the p, d, and q parameters.
2.  The model is prepared on the training data by calling the **fit()** function.
3.  Predictions can be made by using the **predict()** function and specifying the index of the time or times to be predicted.

Now let’s look at an example,

We are going to use a dataset called ‘Shampoo sales’

![](https://miro.medium.com/v2/resize:fit:816/1*_aO9qT4ba-DoULv3TTiGAw.jpeg)

Picture 9.1

![](https://miro.medium.com/v2/resize:fit:816/1*5gnH1FzifOmV7ALXKdok_g.jpeg)

Picture 9.2

## ACF and PACF

We can calculate the correlation for time-series observations with observations from previous time steps, called _lags_. Since the correlation of the time series observations is calculated with values of the same series at previous times, this is called a serial correlation, or an autocorrelation.

A plot of the autocorrelation of a dataset of a time series by lag is called the **A**uto**C**orrelation **F**unction, or the acronym **ACF**. This plot is sometimes called a _correlogram_ or an _autocorrelation plot._

For example,

![](https://miro.medium.com/v2/resize:fit:816/1*072TV36KQtyTDwOKIHCrTA.jpeg)

Picture 10

A partial autocorrelation or **_PACF_** is a summary of the relationship between an observation in a time series with observations at prior time steps with the relationships of in between observations removed.

For example,

![](https://miro.medium.com/v2/resize:fit:816/1*zoL3DJEH9C2-NEPt01snsQ.jpeg)

Picture 11

## Conclusion

Time series analysis is one of the most important aspect of data analytics for any large organization as it helps in understanding seasonality, trends, cyclicality and randomness in the sales and distribution and other attributes. These factors help companies in making a well informed decision which is highly crucial for business.
