# Introduction to Lists in Python

## 1. Understanding Lists
Lists in Python are ordered collections of items that can store multiple types of data such as numbers, strings, and even other lists. Lists are defined using square brackets `[]`.

## 2. Creating and Accessing Lists
```python
# Creating a list of students
students = ["Alice", "Bob", "Charlie", "David", "Eva"]

# Creating a list of student marks
math_marks = [85, 90, 78, 92, 88]

# Accessing elements using index
print(f"First student: {students[0]}")  # Alice
print(f"Last student's mark: {math_marks[-1]}")  # 88
```

## 3. Modifying Lists
Lists in Python are mutable, which means you can change their elements after they are created.
```python
# Adding a new student
students.append("Frank")
math_marks.append(80)  # Adding Frank's mark

# Removing a student
students.remove("Charlie")
math_marks.pop(2)  # Removing Charlie’s mark

# Updating an element
math_marks[1] = 95  # Updating Bob’s mark

print(f"Updated Student List: {students}")
print(f"Updated Marks: {math_marks}")
```

## 4. Slicing Lists
Slicing allows you to access a subset of elements from a list.
```python
# First three students
print(f"First three students: {students[:3]}")

# Last two students
print(f"Last two students: {students[-2:]}")

# Marks of first three students
print(f"Marks of first three students: {math_marks[:3]}")
```

## 5. List Operations
Lists support various operations such as concatenation and repetition.
```python
# Concatenation
extra_students = ["Grace", "Hannah"]
all_students = students + extra_students
print(f"All Students: {all_students}")

# Repetition
repeated_marks = math_marks * 2
print(f"Repeated Marks: {repeated_marks}")
```

## 6. Iterating Through Lists
You can use loops to iterate through a list.
```python
for student in students:
    print(f"Student Name: {student}")
```

## 7. List Comprehension
List comprehension provides a concise way to create lists.
```python
# Creating a list of squared numbers
squared_marks = [mark**2 for mark in math_marks]
print(f"Squared Marks: {squared_marks}")
```

## 8. Sorting and Reversing Lists
```python
# Sorting marks in ascending order
math_marks.sort()
print(f"Sorted Marks: {math_marks}")

# Reversing the list
math_marks.reverse()
print(f"Reversed Marks: {math_marks}")
```

## Class Exercises
- **Task 1:** Create a list of 5 subjects and their marks.
- **Task 2:** Find the highest and lowest marks from the list.
- **Task 3:** Slice and print the first three subjects.
- **Task 4:** Create a list comprehension to increase each mark by 5.

This structured lesson will help you understand lists in Python with practical, class-related examples.
