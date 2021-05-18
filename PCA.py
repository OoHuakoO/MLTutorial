import numpy as np
import pandas as pd
from numpy import *
from sklearn.decomposition import PCA

# Suppose, we have 3 features, X = {X1, X2, X3}, which represents the value of 5 sample
X = np.matrix([[90, 60, 90],
               [90, 90, 30],
               [60, 60, 60],
               [60, 60, 90],
               [30, 30, 30]])

df = pd.DataFrame(X, columns=['column1', 'column2', 'column3'])
print(df)
print("\n")

# calculate the mean of each column
X_mean = mean(X.T, axis=1)
print("Mean: ")
print(X_mean)
print("\n")

# Compute the covariance matrix of data X:
cov_matrix = cov(X.T, bias=1)
print("Covariance matrix: ")
print(cov_matrix)
print("\n")

# Compute the Principal Components using Eigenvectors:
eigenvalue, eigen_vectors = np.linalg.eig(cov_matrix)

# compute eigenvalue:
print("Eigenvalue: ")
print(eigenvalue)
print("\n")

# compute eigenvalue:
print("Eigen vectors: ")
eigen_vectors = pd.DataFrame(eigen_vectors, columns=['v1', 'v2', 'v3'])
print(eigen_vectors)
print("\n")
