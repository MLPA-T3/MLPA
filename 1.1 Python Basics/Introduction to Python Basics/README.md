# Introduction to Python Basics for Data Analysis


Python is one of the most popular programming languages for data analysis, thanks to its simplicity, readability, and powerful libraries. If you’re new to Python and aiming to use it for data analysis, this guide will help you get started with the fundamentals. We’ll focus on basic Python commands that form the foundation for any data analysis work, using tools like Google Colab or Jupyter Notebook to run your code.

## Why Python for Data Analysis?

Python is widely used in the data science community for several reasons:

- **Ease of Learning**: Python’s syntax is straightforward, making it accessible for beginners.
- **Versatility**: Python can be used for various tasks, from simple scripting to complex machine learning algorithms.
- **Community and Support**: Python has an extensive community, offering numerous resources, libraries, and frameworks.

## Setting Up Your Python Environment

To start writing and running Python code, you can use either Google Colab or Jupyter Notebook. Both are great tools that provide an interactive environment for coding.

### Google Colab

Google Colab is a free, cloud-based service that allows you to write and execute Python code in your browser. It’s convenient for beginners because it requires no setup and provides access to GPUs for more advanced tasks.

- **How to Start**: Visit Google Colab at [colab.research.google.com](https://colab.research.google.com/), sign in with your Google account, and create a new notebook.

![Google Colab](https://miro.medium.com/v2/resize:fit:816/1*x3hpXF14UmyhWcwGUHolow.png)

### Jupyter Notebook

Jupyter Notebook is an open-source web application that allows you to create and share documents with live code, equations, visualizations, and narrative text.

- **How to Start**: If you’ve installed Anaconda, launch Jupyter Notebook from the Anaconda Navigator. If not, install it via pip:

  ```bash
  pip install jupyter
  ```

- Start Jupyter by running:

  ```bash
  jupyter notebook
  ```

## Python Basics for Beginners

Before diving into data analysis, it’s crucial to understand the basic building blocks of Python.

### 1. Python Syntax and Variables

Python’s syntax is clean and easy to understand. Here’s how you can declare variables and perform basic operations:

```python
a = 10
b = 5
sum_result = a + b
difference = a - b
product = a * b
quotient = a / b
print("Sum:", sum_result)
print("Difference:", difference)
print("Product:", product)
print("Quotient:", quotient)
```

![Basic operations](https://miro.medium.com/v2/resize:fit:816/1*YMDc3keYULf1WkTCp1TrXg.png)

### 2. Data Types

Python supports various data types, which are essential for handling data. Some of the most common ones include:

- **Integers and Floats**: For numerical data.

  ```python
  x = 5
  ```

- **Strings**: For text data.

  ```python
  name = "Alice"
  greeting = "Hello, " + name
  print(greeting)
  ```

### 3. Control Flow Statements

Control flow statements allow you to execute code based on certain conditions or repeatedly.

- **If-Else Statements**: Used to execute code only if a certain condition is met.

  ```python
  age = 20
  if age >= 18:
      print("You are an adult.")
  else:
      print("You are a minor.")
  ```

- **For Loops**: Used to iterate over a sequence (like a list or a range).

  ```python
  for i in range(5):
      print(i)
  ```

- **While Loops**: Continue executing as long as a condition is true.

  ```python
  count = 0
  while count < 5:
      print(count)
      count += 1
  ```

### 4. Functions

Functions allow you to write reusable pieces of code. They take inputs, process them, and return an output.

- **Defining a Function**:

  ```python
  def add_numbers(a, b):
      return a + b

  result = add_numbers(3, 7)
  print(result)
  ```

- **Using Default Arguments**:

  ```python
  def greet(name="Guest"):
      print("Hello, " + name)

  greet("Alice")
  greet()
  ```

## Collections Data Types in Python

Python offers a variety of collection data types that allow you to store and manipulate groups of data. Here’s a look at the most commonly used collections:

### 1. Lists

Lists are ordered, mutable collections of items. You can store items of any data type, and they can be accessed using their index.

- **Creating a List**:

  ```python
  fruits = ["apple", "banana", "cherry"]
  print(fruits)  # Output: ['apple', 'banana', 'cherry']
  ```

- **Accessing List Items**:

  ```python
  print(fruits[1])
  ```

- **Modifying List Items**:

  ```python
  fruits[1] = "blueberry"
  print(fruits)  # Output: ['apple', 'blueberry', 'cherry']
  ```

- **Looping Through a List**:

  ```python
  for fruit in fruits:
      print(fruit)
  ```

- **List Comprehensions**:

  ```python
  squares = [x**2 for x in range(5)]
  print(squares)
  ```

### 2. Tuples

Tuples are ordered, immutable collections. Once a tuple is created, you cannot change its values. They are often used to store multiple items in a single variable.

- **Creating a Tuple**:

  ```python
  dimensions = (1920, 1080)
  print(dimensions)
  ```

- **Accessing Tuple Items**:

  ```python
  width = dimensions[0]
  height = dimensions[1]
  print(f"Width: {width}, Height: {height}")
  ```

### 3. Dictionaries

Dictionaries are unordered collections of key-value pairs. Each key is unique, and you use the key to access its corresponding value.

- **Creating a Dictionary**:

  ```python
  student = {"name": "John", "age": 22, "major": "Computer Science"}
  print(student)
  ```

- **Accessing Dictionary Items**:

  ```python
  name = student["name"]
  print(name)
  ```

- **Adding/Modifying Dictionary Items**:

  ```python
  student["age"] = 23  # Modifies existing item
  student["grade"] = "A"  # Adds new item
  print(student)
  ```

- **Looping Through a Dictionary**:

  ```python
  for key, value in student.items():
      print(f"{key}: {value}")
  ```

### 4. Sets

Sets are unordered collections of unique items. They are useful when you want to eliminate duplicates or perform mathematical operations like unions and intersections.

- **Creating a Set**:

  ```python
  colors = {"red", "green", "blue"}
  print(colors)
  ```

- **Adding/Removing Items**:

  ```python
  colors.add("yellow")
  colors.remove("green")
  print(colors)
  ```

- **Set Operations**:

  ```python
  a = {1, 2, 3, 4}
  b = {3, 4, 5, 6}

  union = a.union(b)
  intersection = a.intersection(b)
  difference = a.difference(b)

  print("Union:", union)
  print("Intersection:", intersection)
  print("Difference:", difference)
  ```

## Loading and Reading CSV Files

A common task in data analysis is loading data from files, especially CSV (Comma-Separated Values) files. Python’s built-in capabilities allow you to handle CSV files easily.

### Using the CSV Module

The `csv` module in Python provides functionality to both read from and write to CSV files.

- **Reading a CSV File**:

  ```python
  import csv

  with open('data.csv', mode='r') as file:
      csv_reader = csv.reader(file)
      for row in csv_reader:
          print(row)
  ```

### Reading CSV Files Using `pandas`

While this phase focuses on basic Python commands, it’s worth noting that `pandas` is a powerful library
