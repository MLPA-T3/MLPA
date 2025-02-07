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

# Creating a proper 3D array (3x3x3 Tensor)
arr_3d = np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
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

## 8. Joining and Splitting Arrays
### Join arrays along an existing axis
```python
# Creating two-dimensional arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6]])

# Concatenating along axis 0 (rows)
concatenated_rows = np.concatenate((array1, array2), axis=0)
print("Concatenated along rows:\n", concatenated_rows)

# Concatenating along axis 1 (columns)
concatenated_columns = np.concatenate((array1, array2.T), axis=1)
print("\nConcatenated along columns:\n", concatenated_columns)
```

### Splitting an array into multiple subarrays
```python
# Creating an array
original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Splitting the array into three subarrays
subarrays = np.split(original_array, [3, 7])

# Displaying the result
for i, subarray in enumerate(subarrays):
    print(f"Subarray {i + 1}: {subarray}")
```

## 9. Mathematical Operations
### Element-wise operations, Aggregate Functions, and Array Statistics
```python
# Element-wise operations
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
addition = np.add(a, b)
multiplication = np.multiply(a, b)

# Aggregate functions
sum_a = np.sum(a)

# Array statistics
mean_b = np.mean(b)
min_a = np.min(a)
max_b = np.max(b)

print("Element-wise Addition:", addition)
print("Element-wise Multiplication:", multiplication)
print("Sum of Array 'a':", sum_a)
print("Mean of Array 'b':", mean_b)
print("Minimum of Array 'a':", min_a)
print("Maximum of Array 'b':", max_b)
```

## Class Exercises
- **Task 1:** Create a 1D array of 10 elements using `arange()`.
- **Task 2:** Convert a 1D array into a 2D array.
- **Task 3:** Slice and extract the first three elements from a 2D array.
- **Task 4:** Compute the mean, median, and standard deviation of an array.
- **Task 5:** Concatenate two 2D arrays along both axes.
- **Task 6:** Split an array into multiple parts.
