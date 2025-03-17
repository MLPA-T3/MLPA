
## A Comprehensive Guide to Pandas for Data Science

## Pandas

Pandas is an open-source python package that provides numerous tools for high-performance data analysis and data manipulation.

Let’s learn about the most widely used Pandas Library in this article.

## Table of content

1.  Pandas Series
2.  Pandas DataFrame
3.  How to create Pandas DataFrame?
4.  Understanding Pandas DataFrames
5.  Sorting Pandas DataFrames
6.  Indexing and Slicing Pandas Dataframes
7.  Subset DataFrames based on certain conditions
8.  How to fill/drop the null values?
9.  Lambda functions to modify dataframe
10.  Merge, Concatenate dataframes
11.  Grouping and aggregating

## Pandas Datastructures

Pandas supports two datastructures

1.  Pandas Series
2.  Pandas DataFrame

## Pandas Series

Pandas Series is a one-dimensional labeled array capable of holding any data type. Pandas Series is built on top of NumPy array objects.

![](https://miro.medium.com/v2/resize:fit:647/1*VEpxTmtGvXCJfjAHQbqjsQ.png)

Pandas Series having default indexing

In Pandas Series, we can mention index labels. If not provided, by default it will take default indexing(RangeIndex `0 to n-1` )

![](https://miro.medium.com/v2/resize:fit:653/1*2FW6geb3I7veraXk9-42Fw.png)

Index labels have been provided.

## **Accessing elements from Series.**

1.  For Series having default indexing, same as python indexing

![](https://miro.medium.com/v2/resize:fit:416/1*az0F1avI3BUHN0gpVFpaIA.png)

2\. For Series having access labels, same as python dictionary indexing.

![](https://miro.medium.com/v2/resize:fit:458/1*nBZecSou3Ua3hi_Tn9KugQ.png)

## Pandas DataFrame

Pandas Dataframe is a two dimensional labeled data structure. It consists of rows and columns.

Each column in Pandas DataFrame is a Pandas Series.

## How to Create Pandas DataFrames?

We can create pandas dataframe from dictionaries,json objects,csv file etc.

1.  **From csv file**

![](https://miro.medium.com/v2/resize:fit:816/1*Jo1CCMVfIHQGsLE-rMaTsQ.png)

2\. **From a dictionary**

![](https://miro.medium.com/v2/resize:fit:647/1*NyCmcekHPXfp-Fx6kwJ2sQ.png)

**3.From JSON object**

![](https://miro.medium.com/v2/resize:fit:816/1*lv9FXaf7SXwFGP9PGQvopw.png)

## Understanding Pandas DataFrames

![](https://miro.medium.com/v2/resize:fit:247/1*iSV8yuR_1BBwgbmpHyD3iA.png)

df

1.  **df.head()** →Returns first 5 rows of dataframe (by default). Otherwise, it returns the first ’n’ rows mentioned.

![](https://miro.medium.com/v2/resize:fit:431/1*Gavc-c2cP-Mqh9nMDSmIXw.png)

Returns the first row of the dataframe

2\. **df. tail()** →Returns the last 5 rows of the dataframe(by default). Otherwise it returns the last ’n’ rows mentioned.

![](https://miro.medium.com/v2/resize:fit:416/1*v0lcBuEYl9Q3zFwapr_4jQ.png)

Returns the last row of the dataframe

3.**df.shape** → Return the number of rows and columns of the dataframe.

![](https://miro.medium.com/v2/resize:fit:234/1*0VtlXIWT8oYgdYysMk5GNg.png)

Returns the (no of rows, no of columns) of the dataframe

4.**df.info()** →It prints the concise summary of the dataframe. This method prints information of the dataframe like column names, its datatypes, nonnull values, and memory usage

![](https://miro.medium.com/v2/resize:fit:634/1*VJZSisPUnFdHl75zCFgtYw.png)

Summary of the dataframe

5. **df.dtypes()** → Returns a series with the datatypes of each column in the dataframe.

![](https://miro.medium.com/v2/resize:fit:448/1*M2R5dPqkiLHyFE7_cMYjog.png)

datatypes of the columns

6\. **df. values →** Return the NumPy representation of the DataFrame.  
**df.to\_numpy() →** This also returns the NumPy representation of the dataframe.

![](https://miro.medium.com/v2/resize:fit:663/1*nj790_0xhIzR7bWyEte_5w.png)

values in the dataframe, axes labels will be removed.

7.**df.columns →** Return the column labels of the dataframe

![](https://miro.medium.com/v2/resize:fit:750/1*sag162L_FArHjzy4lwJCdQ.png)

8\. **df. describe() →** Generates descriptive statistics. It describes the summary of all numerical columns in the dataframe.  
**df. describe(include=” all”)** → It describes the summary of all columns in the dataframe.

![](https://miro.medium.com/v2/resize:fit:364/1*hUrvXiuil5sx8MdHGjcKsQ.png)

descriptive statistics of all numerical columns in the df

![](https://miro.medium.com/v2/resize:fit:575/1*utesWtMdFV-ESLRIo5Pvqw.png)

descriptive statistics of all columns in the df

8\. **df.set\_index()** → sets the dataframe index using the existing columns. By default it will have RangeIndex (0 to n-1)

`df.set_index(“Fruits_name”,inplace=True)`  
or  
`df=df.set_index(“Fruits_name”)`

\[To modify the df, have to mention **inplace=True** or have to assign to **df** itself. If not, it will return a new dataframe and the original **df** is not modified..\]

![](https://miro.medium.com/v2/resize:fit:701/1*RZ9-13iiO7G6XiaImbyFdA.png)

“Fruits\_name” → column is set as index

9\. **df.reset\_index()** → Reset the index of the dataframe and use the default -index.

![](https://miro.medium.com/v2/resize:fit:591/1*A6zRlScM1x2m669sUlVokQ.png)

Now df having default index \[0 to n-1\]

10\. **df.col\_name.unique()** → Returns the unique values in the column as a NumPy array.

![](https://miro.medium.com/v2/resize:fit:816/1*eATR_5TyZgzQYSumPfMhCw.png)

unique values in the column ”In\_stock”

11\. **df.col\_name.value\_counts()** → Return a Series containing counts of unique values.  
Suppose if we want to find the frequencies of the values present in the columns, this function is used.

![](https://miro.medium.com/v2/resize:fit:816/1*Q84CiUid_RcKBnqb1zDg8g.png)

11\. **df.col\_name.astype() →** Converting datatype of a particular column.

`df.Price.astype(“int32”)` → Converting data type of “Price” column to int

![](https://miro.medium.com/v2/resize:fit:758/1*CwyUlalO_PuvCkHECYSFhw.png)

## Indexing and Slicing Pandas DataFrame

1.  Standard Indexing
2.  Using iloc → Position based Indexing
3.  Using loc → Label based Indexing

![](https://miro.medium.com/v2/resize:fit:718/1*azVYzlLjnpaMHumn3lsCwQ.png)

Understanding index values and index position \[Image by Author\]

## Standard Indexing

1.  **Selecting rows**

Selecting rows can be done by giving a slice of row index labels or a slice of row index position.

`df[start:stop]`

**start,stop** → it can be row\_index \_position or row\_index \_labels.

**Slice of row index position \[End index is exclusive\]**

df\[0:2\] → Same as python slicing. Returns row 0 till row 1.  
df\[0:6:2\] → Returns row 0 till row 5 with step 2. \[Every alternate rows\]

**Slice of row index values \[End index is inclusive\]**

df\[“Apple”:\] →Returns row “Apple” till the last row in the dataframe  
df\[“Apple”:” Banana”\] → Returns row “Apple” till row “Banana”.

![](https://miro.medium.com/v2/resize:fit:816/1*JGM9EhNt7WngVjrNQeK-mQ.png)

Selecting rows by index position and index values

**Note:**

-   We can’t explicitly mention row index position or row index labels. It will raise a keyError.  
    `df[“Banana”]   df[1]   `Both will raise KeyError.
-   We have to mention only the slice of row index labels/row index position.
-   Selecting rows will return a dataframe.

## 2\. Selecting columns

We can select a single column in two ways. Selecting a single column will return a **series**.  
1.`df[“column_name”]`  
2\. `df.column_name`

If a single column\_name is given inside a list, it will return a dataframe.  
`df[[“column_name”]]`

To select multiple columns, have to mention a list of column\_names

`df[[“column_name1”,”column_name2"]]`

Selecting multiple columns will return a **dataframe**.

![](https://miro.medium.com/v2/resize:fit:816/1*syG9w3jDrEfUQzTOkxCPKA.png)

Selecting single and multiple columns

## 3\. Selecting rows and columns

Selecting rows and columns can be given by

`df[start:stop][“col_name”]`

`df[start:stop][[“col_name”]]`

If we mention a single column, it will return a series.  
If we mention single column/multiple columns in a list, it will return a dataframe.

![](https://miro.medium.com/v2/resize:fit:816/1*gLB5yJiWB8HrrXTRdAlGTg.png)

Selecting a subset of DataFrame

## Using iloc -Integer based Indexing.

Using iloc, we can index dataframes using index position

`df[row_index_pos,col_index_pos]`

**row\_index\_pos** → It can be a single row\_index position, a slice of row index\_ position, list of row\_index\_position.  
This field is mandatory

**col\_index\_pos** → It can be a single col\_index\_position, slice of col\_index\_position or list of col\_index\_position.  
This field is optional. If not provided, by default, it takes all columns.

**1.Selecting rows**

![](https://miro.medium.com/v2/resize:fit:816/1*4WUBRRyFYfFIoATbCkLPgw.png)

Selecting rows using iloc

**2\. Selecting columns**

![](https://miro.medium.com/v2/resize:fit:816/1*bJsVVXZ_qPZMJ1fCKvlHhw.png)

Selecting columns, subsets using iloc

## Using loc

Using loc, we can index dataframes using labels.

`df.loc[row_index_labels,col_index_labels]`

**row\_index\_labels** can be a single row\_index label, a slice of row\_index label, or a list of row\_index labels.  
This field is mandatory.

**col\_index\_labels** can be a single col\_index label, a slice of col\_index\_label, or a list of col\_index labels.  
This field is optional. If not provided, by default, it takes all columns.

\[If the dataframe has default indexing, then row\_index\_position and row\_index labels will be the same\]

1.  **Dataframe not having default indexing**

![](https://miro.medium.com/v2/resize:fit:816/1*88TWKTQvT6Nc6BhuGGU-AQ.png)

Using loc

2. **Dataframe having default indexing**

![](https://miro.medium.com/v2/resize:fit:816/1*Md3JGc2BM0E62FPyMtKsQw.png)

Using loc

## Subset dataframe based on certain conditions

In Pandas, we can subset dataframe based on certain conditions.

**Example.** Suppose we want to select rows having “Price” > 5  
`df[“Price”]>5` will return a booelan array

![](https://miro.medium.com/v2/resize:fit:816/1*IdbbSaUMZr59as0ChCbH4g.png)

We can pass this boolean array inside df. loc or standard indexing  
\[df. iloc means we have to remember the column\_index position\]

![](https://miro.medium.com/v2/resize:fit:816/1*0WS48Fv0O24QgWG8ODo7XA.png)

Filtering dataframe based on condition

![](https://miro.medium.com/v2/resize:fit:816/1*OC-shh2IgzrMxHM_IiKmXg.png)

Only “Price” column

## How to drop/ fill the null values in the dataframe

Suppose we want to check whether pandas dataframe has null values.

`df.isnull().sum()` → Returns the sum of null values in each column in the df

![](https://miro.medium.com/v2/resize:fit:816/1*Pb5So4xGfuy6NvkFyzaMnA.png)

`df[“col_name”].isnull().sum()` → Returns the sum of null values for that particular column in the df.

![](https://miro.medium.com/v2/resize:fit:816/1*7esPO2L90WQ8EJHyYJxJAw.png)

If we want to look into the rows which have null values

`df[df[“Price”].isnull()]` → Returns the row which has null values in the “Price” column.

![](https://miro.medium.com/v2/resize:fit:816/1*IYGAoL4Pz5oobmyD2gLVBQ.png)

After looking into the rows, we can decide whether to drop or fill the null values.

If we want to **fill the null values** with a mean value

`df[“col_name”].fillna(df[“col_name”].mean())`

![](https://miro.medium.com/v2/resize:fit:816/1*_1gYYWVPiuZnzL5I1goK8g.png)

If we want to **drop the rows** having a null value.

1.  To drop the rows having null values in a particular column  
    `df.dropna(subset=[“col_name”])`

![](https://miro.medium.com/v2/resize:fit:816/1*Tpijo-asnSFYAulPMpdkCg.png)

Only rows having null values in the “In\_stock” column is removed

2\. To drop all the rows having null values  
`df.dropna()`

![](https://miro.medium.com/v2/resize:fit:816/1*eOlgv6ODo4W6NLgdNFB3VA.png)

All rows having null values are removed.

To modify the original df, have to mention `inplace=True` or have to assign it to the original df itself.

3\. Specify the boolean condition to drop the null values.

![](https://miro.medium.com/v2/resize:fit:816/1*Df3haLVczZHopOkn4MPcUA.png)

## Lambda Functions to modify a column in the dataframe

Suppose if the columns in our dataframe are not in the correct format means, we need to modify the column. We can apply **lambda functions** on a column in a dataframe using the **apply()** method.

**Example 1:** Let’s check the columns and datatypes of the columns in the dataframe (**df.dtypes**).

![](https://miro.medium.com/v2/resize:fit:687/1*tgeOe4a2fZAsvmLXC5OdXw.png)

“Price” column is in **object datatype**. We need to change it to **int data type** so that we can perform mathematical operations.

“Price” column has `$` symbol laso. We need to remove that and then convert it to int datatype.

Have to write lambda functions to remove the `$` sign and to convert it to `int` datatype

`lambda x: int(x[1:])` → This lambda function will remove `$` sign (which is in index 0 in the “Price” column) and convert it to int datatype.

Let’s apply this lambda function using **apply()** method on **“Price”** column

`df3[“Price”].apply(lambda x: int(x[1:]))`

![](https://miro.medium.com/v2/resize:fit:673/1*hTx6cXurZ9lr1BmT1e80FA.png)

To modify the original dataframe, we can assign it to the “Price” column

`df3[“Price”]=df3[“Price”].apply(lambda x: int(x[1:]))`

![](https://miro.medium.com/v2/resize:fit:710/1*LfFmEzN3BcfKzZ8fUMmM2g.png)

“Price” column has been modified.

**Example 2:** If we have null values in the “Price” column and we have replaced that null values with the mean value.

![](https://miro.medium.com/v2/resize:fit:759/1*snsfOAoT5Z3YdF3jqlWaMQ.png)

We can see that the “Price” column having float numbers with many decimal places.

Now, using the lambda function, we can round it to two decimal places.

![](https://miro.medium.com/v2/resize:fit:764/1*XGQxpzRig0G8tvVKBKhLGQ.png)

## Merge, Concat DataFrames

Sometimes, we need to merge/ concatenate multiple dataframes, since data comes in different files.

## Merging Dataframes

`pd.merge()` → Used to merge multiple dataframes using a common column.

**Example.** Let’s see different ways to merge the two dataframes.

![](https://miro.medium.com/v2/resize:fit:640/1*-rcGsuSBhNQrFzgc4j3ngQ.png)

We have “Product\_ID” in common in both dataframes. Let’s merge df1 and df2 on “Product\_Id”.

1.  **inner**

`pd.merge(df1,df2,how=”inner”,on=”Product_ID”)` →It will create a dataframe containing columns from both df1 and df2. Merging happens based on values in column “Product\_ID”

**inner** → similar to an intersection or SQL inner join. It will return only the common rows in both the dataframes.

![](https://miro.medium.com/v2/resize:fit:676/1*_RC1OuRKk3A_v2fMpPWX7Q.png)

Inner merge

2\. **outer**

`pd.merge(df1,df2,how=”outer”,on=”Product_ID”)`

**outer** → Similar to union/SQL full outer join. It will return all the rows from both dataframe.

![](https://miro.medium.com/v2/resize:fit:708/1*kgOcBVjf2vconNRbeT5imQ.png)

Outer Merge

**3\. left**

`pd.merge(df1,df2,how=”left”,on=”Product_ID”)`

**left** → Returns all rows from left df. Here left df = df1

![](https://miro.medium.com/v2/resize:fit:696/1*2oM5x9DsJt3kIlRL0kwiVQ.png)

Left merge

**4\. right**

`pd.merge(df1,df2,how=”right”,on=”Product_ID”)`

**right** → Returns all rows from left df. Here right df = df2

![](https://miro.medium.com/v2/resize:fit:684/1*2wPk0WRhW0ObejstTstzRA.png)

Right merge

## Concatenate Dataframes

`pd.concat()` → It will concatenate two dataframes on top of each other or side by side.

Mostly, this can be used when we want to concatenate two dataframes having the same columns

**Example:**

`pd.concat([df1,df2])` → It will concatenate df1 and df2 on top of each other.  
`pd.concat([df1,df2],ignore_index=True)` → If we want to ignore the index column, set ignore\_index=True.

![](https://miro.medium.com/v2/resize:fit:788/1*7h3qBTJYhGWQM3lFDZI3mA.png)

## Grouping and Aggregating

In pandas, groupby operation involves three steps.

1.  Splitting the data into groups based on some criteria.
2.  Applying a function to each group( Ex. sum(),count(),mean() ..)
3.  Combines the result into a data structure like DataFrame.

**Example:** We want to calculate the total profit of each dept.

![](https://miro.medium.com/v2/resize:fit:289/1*RRmMZ47StuXwIJXFxzUJlw.png)

Dataframe

First, we have to do groupby() on Dept column.  
Then we have to apply the aggregate function → sum() on that group

1.  **Creating groupby object on “Dept”**

Typically, we will group the data using a categorical variable.

```
<span id="92e5" data-selectable-paragraph="">dept_grp=df1.groupby(<strong>"Dept"</strong>)<br>dept_grp</span>
```

**Output**: `<pandas.core.groupby.generic.DataFrameGroupBy object at 0x048DA238>`

df.groupby() returns a groupby object.

2\. **Applying aggregate function on groupby object.**

The aggregation function returns a single aggregate value for each of the groups.  
In our example, we have “Electronics”,” Furniture” and “School Supplies” groups.

![](https://miro.medium.com/v2/resize:fit:607/1*TJcswLEz3Iec8w2kWrqnSg.png)

Total Profit on each Dept

![](https://miro.medium.com/v2/resize:fit:607/1*6DxpB130-dlCZzVdSJ1vWA.png)

Mean Profit on each Dept

**3\. Combine the results into a DataStructure**

![](https://miro.medium.com/v2/resize:fit:640/1*Ak3slPpnqdCSifkmZ6Yu8Q.png)

