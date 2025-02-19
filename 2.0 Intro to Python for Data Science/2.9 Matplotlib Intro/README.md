
# **What is Matplotlib?**

Matplotlib is one of the most widely used data visualization libraries in Python. It’s powerful, flexible, and easy to use, making it an essential tool for anyone working with data. In this blog, we will take a journey from the basics to advanced functionalities of Matplotlib. Whether you are a beginner or looking to polish your skills, this guide has something for everyone.

Before using matplotlib, it is important to have it installed in your system. You can do it by typing the following command in your command prompt:

```
pip install matplotlib
```

To get started, you need to import the library. The standard way to import Matplotlib is:

```
import matplotlib.pyplot as plt
import numpy as np
```

Here `plt` is the alias that we have given so that it is easy to call the library when we need to use it.

We are also importing `numpy` here which is another Python library that is used to work with arrays. You can read more about it [here](https://medium.com/@jainvidip/dance-with-the-arrays-a-complete-guide-to-numpy-00160d4d648e).

Matplotlib allows us to plot various types of plots. We will go through them one by one.

## **Line Plot**

-   **Description:** Creates a line plot.
-   **Example:** `plt.plot(x, y)` plots the values of `y` against `x` as a continuous line.

```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Line Plot](https://miro.medium.com/v2/resize:fit:816/1*jZtLmIWhkfFHU1ikDHb4eQ.png)

## **Scatter Plot**

-   **Description:** Creates a scatter plot.
-   **Example:** `plt.scatter(x, y)` plots the values of `y` against `x` as individual points.

```
import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Scatter Plot](https://miro.medium.com/v2/resize:fit:816/1*QWFj_-cDLSnv5MHIa7u7lA.png)

## **Bar Plot**

-   **Description:** Creates a bar plot.
-   **Example:** `plt.bar(categories, values)` plots bars for each category with the corresponding values.

```
import matplotlib.pyplot as plt
import numpy as np

categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 4]

plt.bar(categories, values)
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()
```

**Output**

![Bar Plot](https://miro.medium.com/v2/resize:fit:816/1*dIMuN_ou7JCgOpcVVXJ6ng.png)

## **Basic Plot Customization**

You may feel like these plots look very basic and dull, well you are right but Matplotlib allows various customization options as well to make our graphs come to life.

### **Legend()**

Places a legend on the axis

```
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title("Sine and Cosine Waves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

**Output**

![Sine and Cosine Plot](https://miro.medium.com/v2/resize:fit:816/1*yA5hf7KRLLSyYek8euYg-A.png)

## **Line Styles and Colors**

-   `color`: Sets the line color  
-   `linestyle`: Sets the line style  
-   `linewidth`: Sets the line width  

```
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='blue', linestyle='--', linewidth=2)
plt.title("Styled Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Styled Line Plot](https://miro.medium.com/v2/resize:fit:816/1*UwyUUQyF66IEwlg2tfcD7g.png)


# **Markers**

We can also add markers to our plots:

- `marker`: Sets the marker style  
- `markersize`: Sets the marker size  
- `markerfacecolor`: Sets the marker face color  
- `markeredgecolor`: Sets the marker edge color  

```
x = np.linspace(0, 10, 10)
y = np.sin(x)
plt.plot(x, y, marker='o', markersize=10, markerfacecolor='red', markeredgecolor='black')
plt.title("Line Plot with Markers")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Line Plot with Markers](https://miro.medium.com/v2/resize:fit:816/1*mnJNvc_j06RSKkvQqOOcfQ.png)

# **Grid**

- `plt.grid()`: Adds grid lines to the plot.

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Plot with Grid Lines")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
```

**Output**

![Plot with Grid Lines](https://miro.medium.com/v2/resize:fit:816/1*DJgUZ8Hp4B6dq8yOq2Gu1A.png)

# **Subplots**

- `plt.subplots()`: Creates multiple plots in a single figure.

```
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y1, 'r')
axs[0].set_title('Sin(x)')
axs[1].plot(x, y2, 'b')
axs[1].set_title('Cos(x)')
plt.tight_layout()
plt.show()
```

**Output**

![Subplots](https://miro.medium.com/v2/resize:fit:816/1*efGtiOYXYbvEC7HJOhs9PA.png)

# **Annotations**

`plt.annotate()`: Adds annotations to the plot.

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Plot with Annotation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.annotate('Max', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

**Output**

![Plot with Annotation](https://miro.medium.com/v2/resize:fit:816/1*FRlpKj4Cpsg0ELzE1cAXVA.png)

# **Logarithmic Scale**

Using Matplotlib, we can set our scale to logarithmic.

- `plt.yscale()`: Sets the y-axis to a logarithmic scale.  
- `plt.xscale()`: Sets the x-axis to a logarithmic scale.  

```
x = np.linspace(1, 10, 100)
y = np.exp(x)
plt.plot(x, y)
plt.title("Logarithmic Scale")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.yscale('log')
plt.show()
```

**Output**

![Logarithmic Scale](https://miro.medium.com/v2/resize:fit:816/1*ls8WHE3NEusndWMQnJqjeQ.png)

# **Figure Size and Resolution**

We can specify figure size and resolution of our plots.

`plt.figure(figsize=(width, height), dpi=dpi)`: Customizes the size and resolution of the plot.

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 4), dpi=100)
plt.plot(x, y)
plt.title("Custom Figure Size and DPI")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Custom Figure Size and DPI](https://miro.medium.com/v2/resize:fit:816/1*6B2vT4ozIjYa9oduWP1ZAg.png)

# **Color Maps**

We can also add some color to our graphs.

`cmap`: Applies a colormap to the plot.

```
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()
plt.title("Scatter Plot with Colormap")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Scatter Plot with Colormap](https://miro.medium.com/v2/resize:fit:802/1*P8XfkEgJwm1DVkW5GqIUtw.png)

# **Customized Plot**

Now I will display a plot using all the customizations that we used. It might look like a huge code at first but when you focus and try to understand, you will see that we have covered all the topics, and you can generate a similar graph yourself.

```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x, y1, color='blue', linestyle='--', linewidth=2, marker='o', markersize=6, markerfacecolor='red', label='sin(x)')
plt.plot(x, y2, color='green', linestyle='-', linewidth=2, marker='s', markersize=6, markerfacecolor='yellow', label='cos(x)')
plt.title("Customized Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.annotate('Max sin', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Min cos', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2 - 1, -1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

**Output**

![Customized Plot](https://miro.medium.com/v2/resize:fit:816/1*HCDxkHGS_L7cvCsvUqSWug.png)


# **Markers**

We can also add markers to our plots:

- `marker`: Sets the marker style  
- `markersize`: Sets the marker size  
- `markerfacecolor`: Sets the marker face color  
- `markeredgecolor`: Sets the marker edge color  

```
x = np.linspace(0, 10, 10)
y = np.sin(x)
plt.plot(x, y, marker='o', markersize=10, markerfacecolor='red', markeredgecolor='black')
plt.title("Line Plot with Markers")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Line Plot with Markers](https://miro.medium.com/v2/resize:fit:816/1*mnJNvc_j06RSKkvQqOOcfQ.png)

# **Grid**

- `plt.grid()`: Adds grid lines to the plot.

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Plot with Grid Lines")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
```

**Output**

![Plot with Grid Lines](https://miro.medium.com/v2/resize:fit:816/1*DJgUZ8Hp4B6dq8yOq2Gu1A.png)

# **Subplots**

- `plt.subplots()`: Creates multiple plots in a single figure.

```
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y1, 'r')
axs[0].set_title('Sin(x)')
axs[1].plot(x, y2, 'b')
axs[1].set_title('Cos(x)')
plt.tight_layout()
plt.show()
```

**Output**

![Subplots](https://miro.medium.com/v2/resize:fit:816/1*efGtiOYXYbvEC7HJOhs9PA.png)

# **Annotations**

`plt.annotate()`: Adds annotations to the plot.

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Plot with Annotation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.annotate('Max', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

**Output**

![Plot with Annotation](https://miro.medium.com/v2/resize:fit:816/1*FRlpKj4Cpsg0ELzE1cAXVA.png)

# **Logarithmic Scale**

Using Matplotlib, we can set our scale to logarithmic.

- `plt.yscale()`: Sets the y-axis to a logarithmic scale.  
- `plt.xscale()`: Sets the x-axis to a logarithmic scale.  

```
x = np.linspace(1, 10, 100)
y = np.exp(x)
plt.plot(x, y)
plt.title("Logarithmic Scale")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.yscale('log')
plt.show()
```

**Output**

![Logarithmic Scale](https://miro.medium.com/v2/resize:fit:816/1*ls8WHE3NEusndWMQnqjeQ.png)

# **Figure Size and Resolution**

We can specify figure size and resolution of our plots.

`plt.figure(figsize=(width, height), dpi=dpi)`: Customizes the size and resolution of the plot.

```
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 4), dpi=100)
plt.plot(x, y)
plt.title("Custom Figure Size and DPI")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Custom Figure Size and DPI](https://miro.medium.com/v2/resize:fit:816/1*6B2vT4ozIjYa9oduWP1ZAg.png)

# **Color Maps**

We can also add some color to our graphs.

`cmap`: Applies a colormap to the plot.

```
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()
plt.title("Scatter Plot with Colormap")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

**Output**

![Scatter Plot with Colormap](https://miro.medium.com/v2/resize:fit:802/1*P8XfkEgJwm1DVkW5GqIUtw.png)

# **Customized Plot**

Now I will display a plot using all the customizations that we used. It might look like a huge code at first but when you focus and try to understand, you will see that we have covered all the topics, and you can generate a similar graph yourself.

```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x, y1, color='blue', linestyle='--', linewidth=2, marker='o', markersize=6, markerfacecolor='red', label='sin(x)')
plt.plot(x, y2, color='green', linestyle='-', linewidth=2, marker='s', markersize=6, markerfacecolor='yellow', label='cos(x)')
plt.title("Customized Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.annotate('Max sin', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Min cos', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2 - 1, -1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

**Output**

![Customized Plot](https://miro.medium.com/v2/resize:fit:816/1*HCDxkHGS_L7cvCsvUqSWug.png)

# **Pie Chart**

Pie charts are circular statistical graphics, which are divided into slices to illustrate numerical proportions.

```
import matplotlib.pyplot as plt

labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Pie Chart")
plt.show()
```

**Output**

![Pie Chart](https://miro.medium.com/v2/resize:fit:557/1*98rNhg2Q8HqQrd9B8x_goQ.png)

# **Histogram**

We can use `plt.hist()` to plot a histogram.

```
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

**Output**

![Histogram](https://miro.medium.com/v2/resize:fit:816/1*9SPSS1-xJTM7dZAKA3HQ9Q.png)

# **Heatmap**

We can use `plt.imshow()` to plot a heatmap. Heatmaps are extremely useful for exploratory data analysis (EDA).

```
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Heatmap")
plt.show()
```

**Output**

![Heatmap](https://miro.medium.com/v2/resize:fit:728/1*3y8Rop6KZKmfL8ym1CjN6g.png)

# **Box Plots**

Box plots are a great way to visualize the distribution of data and identify outliers. They display the median, quartiles, and potential outliers in your dataset.

We can use `plt.boxplot()` to plot a box plot.

```
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.boxplot(data, vert=True, patch_artist=True, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title("Box Plot")
plt.xlabel("Groups")
plt.ylabel("Values")
plt.show()
```

**Output**

![Box Plot](https://miro.medium.com/v2/resize:fit:816/1*289w-pl2_8G7eZYBoeaqig.png)

# **Saving Plots**

After generating your plots, you should be aware of how to save the generated graphs.

```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Saving Plots")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.savefig('plot.png')
plt.show()
```

**Output**

![Saved Plot](https://miro.medium.com/v2/resize:fit:816/1*y0WYr5G44YuM788R1FrolA.png)

# **%matplotlib inline**

This magic command is specific to Jupyter notebooks. It tells the notebook to display Matplotlib plots inline, directly below the code cells that produce them.

```
%matplotlib inline
```

If you run this command once, then you don't have to use `plt.show()` every time.
