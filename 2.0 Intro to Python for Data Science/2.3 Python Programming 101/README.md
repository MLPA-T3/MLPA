# **Welcome to Machine Learning for Predictive Analytics**
This lesson introduces you to Python basics, which will serve as a foundation for understanding machine learning and predictive analytics.

---

## **1. Print Statements**
### **Basic Syntax**
The `print()` function is used to display output in Python.

```python
print("Welcome to Machine Learning for Predictive Analytics")
```

**Output:**
```
Welcome to Machine Learning for Predictive Analytics
```

---

## **2. Variables and Data Types**
Variables store information that can be used and manipulated in a program. Python supports different data types, including:

- **String**: Text data, enclosed in single or double quotes.
- **Integer**: Whole numbers.
- **Float**: Decimal numbers.
- **Boolean**: Represents `True` or `False`.

### **Examples**
```python
# String
name = "John"

# Integer
age = 33

# Float
height = 5.11

# Boolean
is_student = True

print(f"My name is {name}, I am {age} years old, and my height is {height} feet.")
```

**Output:**
```
My name is John, I am 33 years old, and my height is 5.11 feet.
```

---

## **3. Conditional Statements**
Conditional statements allow you to perform actions based on conditions.

### **Example**
```python
if is_student:
    print(f"{name} is {age} years old, has a height of {height} feet and is currently a student.")
else:
    print(f"{name} is {age} years old, has a height of {height} feet and is currently not a student.")
```

**Output (if `is_student = True`):**
```
John is 33 years old, has a height of 5.11 feet and is currently a student.
```

---

## **4. Arithmetic Operations**
Python supports basic arithmetic operations such as addition, subtraction, multiplication, division, and more.

### **Examples**
```python
a = 10
b = 3

print("Addition:", a + b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Floor Division:", a // b)
print("Modulus:", a % b)
print("Exponentiation:", a ** b)
```

**Output:**
```
Addition: 13
Multiplication: 30
Division: 3.3333333333333335
Floor Division: 3
Modulus: 1
Exponentiation: 1000
```

---

## **5. Taking Input from Users**
The `input()` function allows users to enter data during program execution.

### **Examples**
```python
name = input("What is your name? ")
print("Hello,", name, "welcome to Python!")
```

**Output:**
```
What is your name? John
Hello, John, welcome to Python!
```

```python
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
print("The sum is:", num1 + num2)
```

**Output:**
```
Enter the first number: 10
Enter the second number: 20
The sum is: 30
```

---

## **6. Loops**
Loops are used to repeat a block of code multiple times.

### **For Loop Example**
```python
for i in range(1, 6):
    print(i)
```

**Output:**
```
1
2
3
4
5
```

---

## **7. Lists**
A list is a collection of items that can hold multiple values.

### **Examples**
```python
numbers = [10, 20, 30, 40, 50]
print("Original list:", numbers)
print("First number:", numbers[0])
print("Last number:", numbers[-1])
print("Sum of numbers:", sum(numbers))
```

**Output:**
```
Original list: [10, 20, 30, 40, 50]
First number: 10
Last number: 50
Sum of numbers: 150
```

### **Finding the Length of a List**
```python
numbers = [1, 2, 3, 4, 5]
length = len(numbers)
print("Length of the list:", length)
print("List:", numbers)
print("First Number:", numbers[0])
print("Last Number:", numbers[4])
```

**Output:**
```
Length of the list: 5
List: [1, 2, 3, 4, 5]
First Number: 1
Last Number: 5
```

---

## **Key Takeaways**
- Use `print()` to display output.
- Variables can store different types of data, such as strings, integers, floats, and booleans.
- Conditional statements (`if`, `else`) allow decision-making in programs.
- Arithmetic operators include addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), floor division (`//`), modulus (`%`), and exponentiation (`**`).
- Use `input()` to take user input and `int()` or `float()` to convert it into numbers.
- Loops (`for`, `while`) repeat tasks efficiently.
- Lists are used to store and manipulate collections of data.
- The `len()` function helps determine the size of a list.

This concludes the first lesson on Python basics. Feel free to ask questions or experiment with these concepts in your Python environment!
