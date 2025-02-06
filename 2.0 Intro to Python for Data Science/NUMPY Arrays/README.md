# Introduction to Arrays in Python

## 1. Understanding Arrays
In Python, arrays can be created using the `numpy` library. Arrays provide a way to store multiple values of the same type efficiently.

## 2. Creating and Accessing Arrays
```python
import numpy as np

# Creating a 1D array
arr_1d = np.array([10, 20, 30, 40, 50])
print(f"1D Array: {arr_1d}")

# Creating a 2D array (Matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D Array:\n{arr_2d}")

# Creating a 3D array (Tensor)
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Array:\n{arr_3d}")
```

## 3. Indexing and Slicing Arrays
```python
# Indexing in 1D array
print(f"First element: {arr_1d[0]}")
print(f"Last element: {arr_1d[-1]}")

# Slicing in 1D array
print(f"First three elements: {arr_1d[:3]}")

# Indexing in 2D array
print(f"Element at row 1, column 2: {arr_2d[0, 1]}")

# Slicing in 2D array
print(f"First row: {arr_2d[0, :]}")
print(f"First column: {arr_2d[:, 0]}")

# Indexing in 3D array
print(f"First matrix:\n{arr_3d[0, :, :]}")
print(f"First row of second matrix: {arr_3d[1, 0, :]}")
```

## 4. Using `arange()` to Create Arrays
```python
# Creating an array from 0 to 9
seq_array = np.arange(10)
print(f"Sequence Array: {seq_array}")

# Creating an array with a step size
even_numbers = np.arange(0, 10, 2)
print(f"Even Numbers: {even_numbers}")
```

## 5. Array Operations
```python
# Mean (average) of array elements
mean_value = np.mean(arr_1d)
print(f"Mean: {mean_value}")

# Median of array elements
median_value = np.median(arr_1d)
print(f"Median: {median_value}")

# Standard deviation
std_dev = np.std(arr_1d)
print(f"Standard Deviation: {std_dev}")
```

## 6. Reshaping Arrays
```python
# Reshaping a 1D array into a 2D array
reshaped_array = arr_1d.reshape(5, 1)
print(f"Reshaped Array:\n{reshaped_array}")
```

## 7. Sorting and Reversing Arrays
```python
# Sorting an array
sorted_array = np.sort(arr_1d)
print(f"Sorted Array: {sorted_array}")

# Reversing an array
reversed_array = arr_1d[::-1]
print(f"Reversed Array: {reversed_array}")
```

## Class Exercises
- **Task 1:** Create a 1D array of 10 elements using `arange()`.
- **Task 2:** Convert a 1D array into a 2D array.
- **Task 3:** Slice and extract the first three elements from a 2D array.
- **Task 4:** Compute the mean, median, and standard deviation of an array.

This structured lesson will help you understand arrays in Python with practical, class-related examples.
