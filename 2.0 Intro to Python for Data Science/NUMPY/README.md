## NumPy

-   _NumPy is a general-purpose array-processing library_
-   _It provides a high-performance multidimensional array object and tools for working with these arrays._
-   _It is the fundamental package for scientific computing with Python. It is open-source software._

## Features of NumPy

_NumPy has various features including these important ones_

-   _A powerful N-dimensional array object_
-   _Sophisticated (broadcasting) functions_
-   _Tools for integrating C/C++ and Fortran code_
-   _Useful linear algebra, Fourier transform, and random number capabilities_

![](https://miro.medium.com/v2/resize:fit:686/1*A4hBW672DMqruuIfW1DYgg.png)

## Import Numpy

![](https://miro.medium.com/v2/resize:fit:429/1*xdLwMI3Z7b3dt2XuNowFmQ.png)

## Array

-   _An array is a collection of items stored at contiguous memory locations._
-   _The idea is to store multiple items of the same type together. This makes it easier to calculate the position of each element by simply adding an offset to a base value, i.e., the memory location of the first element of the array (generally denoted by the name of the array)._
-   _It is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers.NumPys main object is the homogeneous multidimensional array._
-   _It is the central data structure of the NumPy library. An array is a grid of values and it contains information about the raw data.One way we can initialize NumPy arrays is from Python lists, using nested lists for two- or higher-dimensional data._

## Properties of array:

-   _1.Homogeneous_
-   _2.Mutable_
-   _3.Element Wise Operation_

## Parameters of Array:

![](https://miro.medium.com/v2/resize:fit:816/1*RfS8q7h_4uBsXqTTTAILEQ.png)

![](https://miro.medium.com/v2/resize:fit:816/1*PHvmT4hEeQclQP7x9XNV8g.png)

## Use

-   _NumPy arrays are faster and more compact than Python lists._
-   _An array consumes less memory and is convenient to use._
-   _NumPy uses much less memory to store data and it provides a mechanism of specifying the data types._
-   _This allows the code to be optimized even further._

## The difference between a Python list and a NumPy array

![](https://miro.medium.com/v2/resize:fit:816/1*XoVB0-vEyRLrvpg_eeI6xg.png)

## List

-   _List can have elements of different data types for example, \[1,3.4, ‘hello’, ‘a@’\]_
-   _Elements of a list are not stored contiguously in memory._
-   _Lists do not support element wise operations, for example, addition, multiplication, etc. because elements may not be of same type_
-   _Lists can contain objects of different datatype that Python must store the type information for every element along with its element value. Thus lists take more space in memory and are less efficient._
-   _List is a part of core Python._

## Array

-   _All elements of an array are of same data type for example, an array of floats may be: \[1.2, 5.4, 2.7\]_
-   _Array elements are stored in contiguous memory locations. This makes operations on arrays faster than lists._
-   _Arrays support element wise operations. For example, if A1 is an array, it is possible to say A1/3 to divide each element of the array by 3._
-   _NumPy array takes up less space in memory as compared to a list because arrays do not require to store datatype of each element separately._
-   _Array (ndarray) is a part of NumPy library._

![](https://miro.medium.com/v2/resize:fit:577/1*6-Q2b55DyFOBc73mVJgE7Q.png)

![](https://miro.medium.com/v2/resize:fit:816/1*50ySZSMHYIEgLNaEMOGyqg.png)

![](https://miro.medium.com/v2/resize:fit:816/1*pRMk8sGbOPnkQT7YXwwkog.png)

![](https://miro.medium.com/v2/resize:fit:419/1*ONK-eAQ_f2ijWWtIfMEsuQ.png)

## N-Dimensional Array

-   _In NumPy, an N-dimensional array is a data structure that can represent arrays with any number of dimensions._
-   _These arrays are also known as tensors. A 1-dimensional array is a vector, a 2-dimensional array is a matrix, and an N-dimensional array extends this concept to represent data in more than two dimensions._
-   _We can create N-dimensional arrays using the numpy.array function or other specialized functions in NumPy._
-   _In general, N-dimensional arrays in NumPy provide a versatile and efficient way to represent and manipulate multi-dimensional data._

## Uses of N-dimensional Arrays in NumPy:

**_Scientific Computing:_**

**_Example_**_:_ _In physics simulations, a 3D array might represent a spatial grid, where each element stores information about a physical quantity (e.g., temperature) at a specific point in space_**_._**

**_Image and Signal Processing:_**

-   **_Example_**_:_ In image processing, a 3D array can represent a color image, where each element corresponds to the intensity of a specific color at a particular pixel location.

**_Machine Learning:_**

-   **_Example_**_:_ In machine learning, a 2D array may represent a dataset, where each row corresponds to an observation, and each column represents a feature. Labels and weights can be stored in additional dimensions.

**_Neural Networks:_**

-   **_Example_**_:_ Convolutional neural networks (CNNs) often deal with 4D arrays representing images and their channels. Each dimension may represent the width, height, color channels, and batch size.

![](https://miro.medium.com/v2/resize:fit:816/1*m6w9RoOMnWxiQLpCLrY3IA.png)

## 1\. One-Dimensional Array (1D Array)

-   _A 1-dimensional array in NumPy is a data structure that represents a sequence of values along a single axis._
-   _It is similar to a regular Python list but with the added benefit of NumPy’s array operations and optimizations._
-   _we can create a 1D array in NumPy using the numpy.array function and passing a list as an argument.NumPy’s 1D arrays provide a foundation for more complex data structures like 2D and 3D arrays._
-   _They offer performance advantages over traditional Python lists, especially when dealing with large datasets and numerical computations._
-   _The ability to perform vectorized operations makes NumPy well-suited for scientific computing and data analysis tasks.1D arrays are used in Vectorized Operations,Mathematical Operations,Signal Processing,Time Series Analysis_

![](https://miro.medium.com/v2/resize:fit:584/1*QmF00suo1yMi_SePp7h6Yg.png)

![](https://miro.medium.com/v2/resize:fit:483/1*GGksy-fNsIoWUJ2rhi-qZw.png)

## 2\. Two-Dimensional Array(2D Array)

-   _A 2-dimensional array in NumPy is a data structure that represents a grid of values, similar to a matrix.It is a fundamental concept in numerical computing and is widely used in various scientific, engineering, and data analysis applications._
-   _We can create a 2D array in NumPy using the numpy.array function and passing a nested list as an argument._
-   _Representing adjacency matrices in graph theory to describe connections between nodes. NumPy provides a powerful and efficient array object that supports a wide range of operations on 2D arrays._
-   _It is a cornerstone library for numerical computing in Python and is widely used in scientific and engineering applications. \* Working with 2D arrays allows for the efficient manipulation and analysis of structured data._
-   _2D arrays are3 used in Networks and Graphs,Data Visualization,Machine Learning,Tabular Data,Image Processing_

![](https://miro.medium.com/v2/resize:fit:658/1*lxsaLoJh7KV_fAm6QseTPA.png)

![](https://miro.medium.com/v2/resize:fit:552/1*EoguCW3f713RLFVCoffZ3w.png)

## 3\. Three-Dimensional Array(3D Array)

-   _A 3-dimensional array in NumPy is a data structure that represents a three-dimensional cube of values.It extends the concept of a 2-dimensional array (matrix) to an additional dimension._
-   _This structure is often used to represent volumetric data or a collection of 2D arrays over time or some other parameter._
-   _When working with 3D arrays in NumPy, you can perform various operations, manipulations, and analyses similar to those performed on 2D arrays._
-   _NumPy provides a rich set of functions for array manipulation, linear algebra, and mathematical operations, making it a powerful tool. 3D arrays are used in Image Processing,Video Processing,Scientific Data,3D Graphics_

![](https://miro.medium.com/v2/resize:fit:802/1*08pm6XVxD60B044_zQzOxg.png)

![](https://miro.medium.com/v2/resize:fit:604/1*humwseouKY3YYWZr7g90uQ.png)

## 3d numpy array visualization

```python
import numpy as np
import plotly.express as px

# Define cities, years, and growth rates
cities = ['City A', 'City B', 'City C', 'City D']
years = [2000, 2005, 2010, 2015, 2020]
growth_rates = [1.02, 1.03, 1.01, 1.05]  # Annual growth rates for each city

# Initialize a 3D NumPy array (4 cities x 5 years x 1 value per cell)
num_cities = len(cities)
num_years = len(years)
population_array = np.zeros((num_cities, num_years))

# Populate the array with population values
for i, growth_rate in enumerate(growth_rates):
    population = 1_000_000  # Starting population for each city
    for j, year in enumerate(years):
        population = population * (growth_rate ** (year - years[0]) / 5)
        population_array[i, j] = population

# Convert the 3D array into 2D arrays suitable for plotting
x = np.repeat(cities, num_years)  # Cities as names (City A, City B, ...)
y = np.tile(years, num_cities)  # Years repeated for each city
z = population_array.flatten()  # Flatten the population values

# Plot using Plotly
fig = px.scatter_3d(
    x=x, y=y, z=z,
    color=x,  # Color by city
    size=z,  # Bubble size proportional to population
    labels={"x": "City", "y": "Year", "z": "Population"},
    title="3D Visualization of Population Growth by City"
)
fig.update_traces(marker=dict(opacity=0.8))  # Adjust marker transparency
fig.show()

# Print the 3D NumPy array
print("3D NumPy Array:\n", population_array)


fig.write_html("3d_scatter_plot.html")
```

## Attributes of NumPy Array

-   _An array is usually a fixed-size container of items of the same type and size._
-   _The number of dimensions and items in an array is defined by its shape._
-   _The shape of an array is a tuple of non-negative integers that specify the sizes of each dimension._
-   _In NumPy, dimensions are called axes._

![](https://miro.medium.com/v2/resize:fit:816/1*A4HAp1EYiNrBYFAC6dXaKg.png)

## i) ndim:

-   _ndarray.ndim gives the number of dimensions of the array as an integer value. Arrays can be 1-D, 2-D or n-D._
-   _NumPy calls the dimensions as axes (plural of axis)._
-   _Thus, a 2-D array has two axes. The row-axis is called axis-0 and the column-axis is called axis-1._
-   _The number of axes is also called the array’s rank._
-   _ndarray.ndim_

![](https://miro.medium.com/v2/resize:fit:602/1*MySgWxgsWudwbTHI_IjlGQ.png)

![](https://miro.medium.com/v2/resize:fit:620/1*fcwL7a0cpDrA7nLks271dA.png)

## (ii) Shape

-   _It gives the sequence of integers indicating the size of the array for each dimension._
-   _ndarray1.shape_

![](https://miro.medium.com/v2/resize:fit:591/1*M8qHK-yuk_ODb5LQQaGd2g.png)

## (iii) Size

-   _It gives the total number of elements of the array._
-   _This is equal to the product of the elements of shape._
-   _ndarray1.size_

![](https://miro.medium.com/v2/resize:fit:623/1*dHEw0EotjWZkrzQIhmCrcw.png)

## (iv) dtype:

-   _ndarray.dtype is the data type of the elements of the array. All the elements of an array are of same data type._
-   _Common data types are int32, int64, float32, float64, U32, etc._
-   _ndarray1.dtype_

![](https://miro.medium.com/v2/resize:fit:816/1*qa0FPG7LmyayvK287sz8Pg.png)

![](https://miro.medium.com/v2/resize:fit:816/1*3ZsgD0uwxaB0yMvYhi7eog.png)

![](https://miro.medium.com/v2/resize:fit:640/1*VvhKLIeh1ph2fcWYmD5FCg.png)

![](https://miro.medium.com/v2/resize:fit:816/1*4-asa_8-IyP6MB_K1PVrzw.png)

![](https://miro.medium.com/v2/resize:fit:683/1*NrtTT8Ep5oGAUhcYa67aBg.png)

## (v) Itemsize

-   _It specifies the size in bytes of each element of the array._
-   _Data type int32 and float32 means each element of the array occupies 32 bits in memory. 8 bits form a byte._
-   _Thus, an array of elements of type int32 has itemsize 32/8=4 bytes._
-   _Likewise, int64/float64 means each item has itemsize 64/8=8 bytes._
-   _ndarray.itemsize_

![](https://miro.medium.com/v2/resize:fit:653/1*WtOHBXtScIsc8WyMjmQB8w.png)

## arange

-   _NumPy arange() is one of the array creation routines based on numerical ranges._
-   _It creates an instance of ndarray with evenly spaced values and returns the reference to it._
-   _We can define the interval of the values contained in an array, space between them, and their type with four parameters of arange()_
-   _The first three parameters determine the range of the values, while the fourth specifies the type of the elements:_
-   _start is the number (integer or decimal) that defines the first value in the array._
-   _stop is the number that defines the end of the array and isn’t included in the array._
-   _step is the number that defines the spacing (difference) between each two consecutive values in the array and defaults to 1_
-   _dtype is the type of the elements of the output array and defaults to None._
-   _step can’t be zero. Otherwise, you’ll get a ZeroDivisionError._
-   _numpy.arange(\[start, \]stop, \[step, \], dtype=None) -> numpy.ndarray_

![](https://miro.medium.com/v2/resize:fit:816/1*zx9rOfXWY8iU4ftgGWtlGw.png)

## Difference between Range and Arange

-   _The main difference between the two is that range is a built-in Python class, while arange() is a function that belongs to a third-party library (NumPy)._

## range and arange() also differ in their return types:

-   _range creates an instance of this class that has the same features as other sequences (like list and tuple), such as membership, concatenation, repetition, slicing, comparison, length check, and more._
-   _arange() returns an instance of NumPy ndarray._

![](https://miro.medium.com/v2/resize:fit:665/1*1Ij0LGU3YcDrBrVgRqd5Yw.png)

## Reshape

-   _Reshaping numpy array simply means changing the shape of the given array, shape basically tells the number of elements and dimension of array, by reshaping an array we can add or remove dimensions or change number of elements in each dimension._
-   _In order to reshape a numpy array we use reshape method with the given array._
-   _Syntax : array.reshape(shape)_
-   _Argument : It take tuple as argument, tuple is the new shape to be formed_
-   _Return : It returns numpy.ndarray_
-   **_Note_** _: We can also use np.reshape(array, shape) command to reshape the array_

## _Reshaping : 1-D to 2D_

-   _We will reshape the 1-D array of shape (1, n) to 2-D array of shape (N, M) here M should be equal to the n/N there for N should be factor of n_

![](https://miro.medium.com/v2/resize:fit:604/1*m5mPwnrU99lpxmaoWV9RUQ.png)

## Indexing arrays

-   _Indexing can be done in numpy by using an array as an index._
-   _In case of slice, a view or shallow copy of the array is returned but in index array a copy of the original array is returned. \* Numpy arrays can be indexed with other arrays or any other sequence with the exception of tuples._

![](https://miro.medium.com/v2/resize:fit:816/1*AeGQndvZ1C6x27IHUR14eg.png)

![](https://miro.medium.com/v2/resize:fit:528/1*yP-Gzpl1hae-9oQyh9tCjg.png)

![](https://miro.medium.com/v2/resize:fit:618/1*nydcibKVSc1E-Z8SSFVorQ.png)

![](https://miro.medium.com/v2/resize:fit:795/1*NwoWmmO-C56vNzR8G78dCg.png)

## Slicing Arrays:

-   _Basic Slicing and indexing : Consider the syntax x\[obj\] where x is the array and obj is the index._
-   _Slice object is the index in case of basic slicing._
-   _Basic slicing occurs when obj is a slice object that is of the form_
-   _start : stop : step_
-   _an integeror a tuple of slice objects and integers_
-   _All arrays generated by basic slicing are always view of the original array._
