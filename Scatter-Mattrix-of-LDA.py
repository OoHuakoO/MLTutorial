import numpy as np
import pandas as pd
from numpy import *
from sklearn.decomposition import PCA

# Suppose, we have 2 classes and 2 features with 5 elements
w1 = np.matrix([[4, 1],
                [2, 4],
                [2, 3],
                [3, 6],
                [4, 4]])

w2 = np.matrix([[9, 10],
                [6, 8],
                [9, 5],
                [8, 7],
                [10, 8]])

df = pd.DataFrame(w1, columns=['column1', 'column2'])
print("w1: ")
print(df)
print("\n")

df2 = pd.DataFrame(w2, columns=['column1', 'column2'])
print("w2: ")
print(df)
print("\n")

# Compute the d-dimensional mean vectors, N=5.
w1_mean = mean(w1.T, axis=1)
print("Mean of w1: ")
print(w1_mean)
print("\n")

w2_mean = mean(w2.T, axis=1)
print("Mean of w2: ")
print(w2_mean)
print("\n")

#  Compute the scatter matrices for 2 features
scat_matrix1 = cov(w1.T, bias=1)
print("S1: ")
print(scat_matrix1)
print("\n")

scat_matrix2 = cov(w2.T, bias=1)
print("S2: ")
print(scat_matrix2)
print("\n")

# Find SB
SubT = (np.subtract(w1_mean, w2_mean).transpose())
SB = ((np.subtract(w1_mean, w2_mean))*SubT)
print('SB: ')
print(SB)
print("\n")

# find SW
SW = scat_matrix1 + scat_matrix2
print('SW: ')
print(SW)
print("\n")

# Compute the eigenvectors and corresponding eigenvalues for the scatter matrices
# find A
A = (np.linalg.inv(SW))*SB
print('A: ')
print(A)
