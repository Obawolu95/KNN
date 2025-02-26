"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
Task
Implement a function which takes as input
   * two data sets 
   * a number for the maximally considered number of nearest neighbors
   
produces as output
   * a matrix with entries u_k(z_i ) for and k = 1,…, k_max, i = 1,…, N_x + N_y

# Function to compute the k-nearest neighbor matrix
def compute_knn_matrix(X, Y, k_max):
    N_x = X.shape[0]
    N_y = Y.shape[0]
    
    # Initialize the nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k_max, algorithm='auto')
    
    # Fit the model on dataset Y (to find nearest neighbors in Y for each point in X)
    nbrs.fit(Y)
    
    # Find the k-nearest neighbors for each point in X
    distances, indices = nbrs.kneighbors(X)
    
    # The result matrix u_matrix will have shape (N_x, k_max)
    u_matrix = np.zeros((N_x, k_max))
    
    # For each point in X, we fill the distances to its k nearest neighbors from Y
    for i in range(N_x):
        u_matrix[i, :] = distances[i, :]
    
    return u_matrix

# Example usage
X = np.random.rand(10, 2)  # Example dataset X with 10 points, 2D space
Y = np.random.rand(10, 2)  # Example dataset Y with 10 points, 2D space
k_max = 3  # Consider up to 3 nearest neighbors
print(X)
print(Y)
# Compute the nearest neighbor matrix
u_matrix = compute_knn_matrix(X, Y, k_max)

# Plotting the distances for each point
plt.figure(figsize=(10, 6))

# Loop over each point in X and plot its distances to the k nearest neighbors
for i in range(u_matrix.shape[0]):
    plt.plot(range(1, k_max + 1), u_matrix[i, :], marker='o', label=f'Point {i+1}')

plt.title('Nearest Neighbor Distances')
plt.xlabel('k (Nearest Neighbors)')
plt.ylabel('Distance')
plt.legend(title='Points')
plt.grid(True)
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt

def k_nearest_neighbors(dataset1, dataset2, k):
    # Compute pairwise distances between points in dataset1 and dataset2
    distances = np.linalg.norm(dataset1[:, np.newaxis] - dataset2, axis=2)
    
    # Get the indices and distances of the k-nearest neighbors for each point in dataset1
    neighbors_indices = np.argsort(distances, axis=1)[:, :k]  # Indices of k nearest neighbors
    neighbors_distances = np.take_along_axis(distances, neighbors_indices, axis=1)  # Corresponding distances
    
    return neighbors_indices, neighbors_distances

# Define two datasets of the same dimension
dataset1 = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # 4 points in 2D
dataset2 = np.array([[5, 6], [6, 7], [7, 8], [8, 9]])  # 4 points in 2D

# Define the number of nearest neighbors
k = 2

# Call the function to get nearest neighbors
indices, distances = k_nearest_neighbors(dataset1, dataset2, k)

# Plotting
plt.figure(figsize=(8, 6))

# Plot dataset1 and dataset2
plt.scatter(dataset1[:, 0], dataset1[:, 1], color='blue', label='Dataset 1', s=100, marker='o')
plt.scatter(dataset2[:, 0], dataset2[:, 1], color='red', label='Dataset 2', s=100, marker='x')

# For each point in dataset1, plot lines to its k nearest neighbors in dataset2
for i in range(dataset1.shape[0]):
    for j in range(k):
        neighbor_index = indices[i, j]
        plt.plot([dataset1[i, 0], dataset2[neighbor_index, 0]], 
                 [dataset1[i, 1], dataset2[neighbor_index, 1]], 
                 color='gray', linestyle='--', alpha=0.7)  # Connecting line between points

# Add labels and a legend
plt.title('k-Nearest Neighbors between Two Datasets (Same Dimension)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
