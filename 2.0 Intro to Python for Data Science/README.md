

## Indexing and Slicing:

In Python, the elements of ordered sequences like strings or lists can be individually accessed through their indices. This can be achieved by providing the numerical index of the element we wish to extract from the sequence. Additionally, Python supports slicing, a characteristic that lets us extract a subset of the original sequence object.

In Python sequence data types, we can access elements by indexing and slicing. Sequence data types include strings, lists, tuples, and range objects.

![](https://miro.medium.com/v2/resize:fit:636/1*eAkb5qZ29PkyIfJ18bQK-A.png)

## What are Indexing and Slicing?

**Indexing:** _Indexing_ is used to obtain individual elements.

**Slicing:** _Slicing_ is used to obtain a sequence of elements.

Indexing and Slicing can be done in Python sequence types like **list, string, tuple, and range** objects.

## Indexing

Indexing starts from `0`. Index `0` represents the first element in the sequence.

Negative indexing starts from `-1`. Index `-1` represents the last element in the sequence.

## Indexing in String

![](https://miro.medium.com/v2/resize:fit:742/0*7klCIkGF_70z0Lb8.png)

### Example of Indexing in Strings

```python
s = "Python"
print(s[0])   # Output: P
print(s[-1])  # Output: n
print(s[-2])  # Output: o
```

## Indexing in List

![](https://miro.medium.com/v2/resize:fit:64/0*EkTKbamiNMyUNwxk)

![](https://miro.medium.com/v2/resize:fit:742/0*Ql2uVyQPWYtSLdmV.png)

### Example of Indexing in List

```python
list_ = [10, 20, 30, 40, 50]
print(list_[0])   # Output: 10
print(list_[-1])  # Output: 50
print(list_[-2])  # Output: 40
```

## Indexing in Tuple and Range Object

### Indexing in Tuple
```python
t = (1, 2, 3, 4, 5)
print(t[0])   # Output: 1
print(t[-1])  # Output: 5
print(t[-2])  # Output: 4
```

### Indexing in Range Object
```python
r = range(5)
print(r[0])   # Output: 0
print(r[-1])  # Output: 4
print(r[-2])  # Output: 3
```

## Slicing:

### **Slicing (Range of Indexes):**

We can specify a range of indexes using the syntax `s[i:j:k]` where:
- `i` is the starting index.
- `j` is the stopping index (exclusive).
- `k` is the step.

## Slicing Strings

Slicing allows you to obtain substrings:

1. `s[1:3]` — Returns elements from the first index to the third index (excluded).

```python
s = "Python"
print(s[1:3])   # Output: yt
```

2. `s[0:3]` — Returns elements from the beginning of the string till the third index (excluded).

```python
s = "Python"
print(s[0:3])   # Output: Pyt
```

### Slice Indices with Defaults

- **Omitted start index:**

  `s[:4]` — Returns elements from the beginning of the string till the fourth index.

  ```python
  s = "Python"
  print(s[:4])   # Output: Pyth
  ```

- **Omitted stop index:**

  `s[2:]` — Returns elements from the second index till the end of the string.

  ```python
  s = "Python"
  print(s[2:])   # Output: thon
  ```

### Using Negative Index

- `s[-2:]` — Returns elements from the second last index till the end of the string.

  ```python
  s = "Python"
  print(s[-2:])  # Output: on
  ```

### Using Step Index

- `s[1:5:2]` — Returns elements from index `1` to `5` (excluded) using a step of `2`.

  ```python
  s = "Python"
  print(s[1:5:2])  # Output: yh
  ```

![](https://miro.medium.com/v2/resize:fit:64/0*Fxjn7Q-Af-msvJkI)

![](https://miro.medium.com/v2/resize:fit:742/0*uG02yvkjjdUxbyiP.png)

Image Source: Author

### Out of Range Index

Out of range indexes are handled gracefully when used for slicing.

```python
s = "Python"
print(s[10:])  # Output: ''
```

## Slicing List:

Slicing a list returns a new list containing the requested elements.

1. **Shallow Copy:**

  `n[:]` — Returns a shallow copy of the list.

  ```python
  n = [0, 1, 2, 3, 4, 5]
  print(n[:])  # Output: [0, 1, 2, 3, 4, 5]
  ```

2. **Specify Start and Stop Indices:**

  `n[1:3]` — Returns a new list containing elements from index `1` to `3` (excluded).

  ```python
  n = [0, 1, 2, 3, 4, 5]
  print(n[1:3])  # Output: [1, 2]
  ```

3. **Omitted Stop Index:**

  `n[1:]` — Returns a new list containing elements from the first index till the end of the list.

  ```python
  n = [0, 1, 2, 3, 4, 5]
  print(n[1:])  # Output: [1, 2, 3, 4, 5]
  ```

4. **Omitted Start Index:**

  `n[:4]` — Returns a new list containing elements from the beginning of the list till the fourth index.

  ```python
  n = [0, 1, 2, 3, 4, 5]
  print(n[:4])  # Output: [0, 1, 2, 3]
  ```

5. **Slicing vs Indexing:**

  - `n[1:2]` — Returns a new list containing an element from index `1`.
  - `n[1]` — Returns the element at index `1`.

  ```python
  n = [0, 1, 2, 3, 4, 5]
  print(n[1:2])  # Output: [1]
  print(n[1])    # Output: 1
  ```

6. **Using Step:**

  `n[1:5:2]` — Returns a new list containing elements from index `1` to `5` (excluded) using a step of `2`.

  ```python
  n = [0, 1, 2, 3, 4, 5]
  print(n[1:5:2])  # Output: [1, 3]
  
