
# Solving Systems of Linear Equations with NumPy - Lab

## Introduction 

Now you've gathered all the required skills needed to solve systems of linear equations. You saw why there was a need to calculate inverses of matrices, followed by matrix multiplication to figure out the values of unknown variables. 

The exercises in this lab present some problems that can be converted into a system of linear equations. 

## Objectives
You will be able to:

- Use matrix algebra and NumPy to solve a system of linear equations given a real-life example 
- Use NumPy's linear algebra solver to solve for systems of linear equations

## Exercise 1

A coffee shop is having a sale on coffee and tea. 

On day 1, 29 bags of coffee and 41 bags of tea were sold, for a total of 490 dollars.

On day 2, they sold 23 bags of coffee and 41 bags of tea, for which customers paid a total of 448 dollars.  

How much does each bag cost?


```python
# Create and solve the relevant system of equations
import numpy as np
a = np.matrix([[29, 41], [23, 41]])
b = np.matrix([490, 448])
# A dot X = B
```


```python
# Describe your result
b.shape, a.shape
```




    ((1, 2), (2, 2))




```python
a_inv = np.linalg.inv(a)
```


```python
a_inv.dot(b.T)
```




    matrix([[7.],
            [7.]])




```python
np.linalg.solve(a, b.T)
```




    matrix([[7.],
            [7.]])



It costs $7 for coffee and tea.

## Exercise 2

The cost of admission to a popular music concert was 162 dollars for 12 children and 3 adults. 

The admission was 122 dollars for 8 children and 3 adults in the same music concert. 

How much was the admission for each child and adult?


```python
# Create and solve the relevant system of equations
a = np.matrix([[12, 3], [8,3]])
b = np.matrix([162, 122])
a_inv = np.linalg.inv(a)
a_inv.dot(b.T)
```




    matrix([[10.],
            [14.]])




```python
Describe your result:
$10 per child
$14 adults

```


      File "<ipython-input-17-65c3d8cd2a19>", line 1
        Describe your result:
                    ^
    SyntaxError: invalid syntax



## Exercise 3

You want to make a soup containing tomatoes, carrots, and onions.

Suppose you don't know the exact mix to put in, but you know there are 7 individual pieces of vegetables, and there are twice as many tomatoes as onions, and that the 7 pieces of vegetables cost 5.25 USD in total. 
You also know that onions cost 0.5 USD each, tomatoes cost 0.75 USD and carrots cost 1.25 USD each.

Create a system of equations to find out exactly how many of each of the vegetables are in your soup.


```python
# Create and solve the relevant system of equations
# 7 total veggies
# X = 2Z
# X +0Y - 2Z = 0
#5.25 total
# o = 0.5
# t = 0.75
# c = 1.25
# Xt + Yc + Zo = 5.25
#Xt + Yc + 2 * X * o = 5.25
# 0.75X + 1.25Y + 0.5Z = 7
# X + Y + Z = 7
a = np.matrix([[0.75,1.25,0.5], [1,0,-2], [1,1,1]])
b = np.matrix([5.25, 0, 7])
a_inv = np.linalg.inv(a)
a_inv.dot(b.T)
```




    matrix([[4.],
            [1.],
            [2.]])




```python
# Describe your result
We need 4 tomatoes, 1 carrot, and 2 onions
```


      File "<ipython-input-20-a5b5073a8be3>", line 2
        We need 4 tomatoes, 1 carrot, and 2 onions
              ^
    SyntaxError: invalid syntax



## Exercise 4

A landlord owns 3 properties: a 1-bedroom, a 2-bedroom, and a 3-bedroom house. 

The total rent he receives is 1240 USD. 

He needs to make some repairs, where those repairs cost 10% of the 1-bedroom house’s rent. The 2-bedroom repairs cost 20% of the 2-bedroom rental price and 30% of the 3-bedroom house's rent for its repairs.  The total repair bill for all three houses was 276 USD. 

The 3-bedroom house's rent is twice the 1-bedroom house’s rent. 

How much is the individual rent for three houses?


```python
# Create and solve the relevant system of equations
# Total rent = 1240
# repairs = 10% of b_1 rent, 20% b_2, 30 of b_3, totalling 276
# b_3 = 2b_1
# 1*b_3 + 1*b_2 + 1*b_1 = 1240
# 0.3*b_3 +0.2b_2 + 0.1*b_1 = 276
# b_3 + 0b_2 - 2b_1 = 0
a = np.matrix([[1,1,1],[0.3,0.2,0.1], [1, 0, -2]])
b = np.matrix([1240, 276, 0])
a_inv = np.linalg.inv(a)
a_inv.dot(b.T)
```




    matrix([[560.],
            [400.],
            [280.]])



## Summary
In this lab, you learned how to use NumPy to solve linear equations by taking inverses and matrix multiplication and also using numpy's `solve()` function. You'll now take these skills forward and see how you can define a simple regression problem using linear algebra and solve it with Numpy. 
