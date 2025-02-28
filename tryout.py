"""import numpy as np
from sklearn.neighbors import NearestNeighbors
from simulatemodesame import *

def find_knn_for_arbitrary_points(combined_data, original_data, k=10, num_points=20):
   
    Find the k-nearest neighbors for a subset of points from the original dataset.
    
    Parameters:
    - combined_data: Combined dataset of original and synthetic data (numpy array).
    - original_data: Original dataset points (numpy array).
    - k: Number of nearest neighbors to consider (default is 10).
    - num_points: Number of arbitrary points from the original dataset to use (default is 20).
    
    Returns:
    - knn_results: A list of dictionaries, each containing the indices of the k nearest neighbors
                   and the corresponding distances for each point.
   
    # Randomly select num_points indices from the original dataset
    selected_indices = np.random.choice(len(original_data), num_points, replace=False)

    # Initialize NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1)  # k+1 because we exclude the point itself
    nn.fit(combined_data)

    # Dictionary to store the results
    knn_results = []

    # For each of the selected points from the original dataset, find k-nearest neighbors
    for idx in selected_indices:
        # Find k-nearest neighbors for the selected point
        distances, indices = nn.kneighbors([original_data[idx]])

        # We exclude the point itself (the first neighbor is always itself)
        neighbors = indices[0][1:]

        # Store the results
        knn_results.append({
            'point_idx': idx,
            'neighbors': neighbors,
            'distances': distances[0][1:]  # Exclude the distance to itself
        })
    
    return knn_results

# Example usage:
combined_data, synthetic_data = generate_data_with_cluster(original_size=100, synthetic_size=100, 
                                                           cluster_ratio=0.01, data_radius=5, 
                                                           cluster_radius=1, min_distance_from_origin=100, 
                                                           random_seed=42)

# Find the 10 nearest neighbors for 20 arbitrary points from the original dataset
knn_results = find_knn_for_arbitrary_points(combined_data, original_data=combined_data[:100], k=10, num_points=20)

# Display the results for the first few points
for result in knn_results:
    print(f"Original Point {result['point_idx']} KNN neighbors:")
    print(f"Neighbors: {result['neighbors']}")
    print(f"Distances: {result['distances']}")
    print("------")
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

# Generate original and synthetic datasets
def generate_data_with_cluster(original_size=100, synthetic_size=100, cluster_ratio=0.05, data_radius=5, cluster_radius=1, min_distance_from_origin=100, random_seed=42):
    np.random.seed(random_seed)
    
    original_center = np.array([0, 0])  # Center for original data
    original_data = original_center + np.random.normal(loc=0, scale=data_radius, size=(original_size, 2))
    #print("original data is :")
    #print(original_data)
    
    synthetic_center = np.array([0, 0])  # Center for synthetic data
    synthetic_data = synthetic_center + np.random.normal(loc=0, scale=data_radius, size=(synthetic_size, 2))
    
    num_clustered = int(synthetic_size * cluster_ratio)
    clustered_indices = np.random.choice(synthetic_size, num_clustered, replace=False)
    
    angle = np.random.uniform(0, 2 * np.pi, num_clustered)
    radius = np.full(num_clustered, min_distance_from_origin)
    x_offset = radius * np.cos(angle)
    y_offset = radius * np.sin(angle)
    synthetic_data[clustered_indices] += np.column_stack((x_offset, y_offset))
    
    cluster_center = np.mean(synthetic_data[clustered_indices], axis=0)
    synthetic_data[clustered_indices] = cluster_center + np.random.normal(loc=0, scale=cluster_radius, size=(num_clustered, 2))
    
    combined_data = np.vstack((original_data, synthetic_data))
    #print(combined_data)
    labels = np.concatenate((np.ones(original_size), np.full(synthetic_size, 2)))
    #print(labels)
    return combined_data, labels

# Calculate k-nearest neighbors
def calculate_knn(combined_data, k=10):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(combined_data)
    
    knn_distances, knn_indices = knn.kneighbors(combined_data)
    #print(knn_distances.shape)
    #print(knn_indices.shape)
    return knn_indices, knn_distances

# Count original vs. synthetic neighbors
def count_original_vs_synthetic_neighbors(knn_indices, labels):
    counts = []
    for i in range(len(knn_indices)):
        neighbors_labels = labels[knn_indices[i]]
        original_count = np.sum(neighbors_labels == 1)
        synthetic_count = np.sum(neighbors_labels == 2)
        counts.append((original_count, synthetic_count))
    return counts

# Calculate the Hypergeometric Distribution
def hypergeometric_distribution(N_x, N_y, k, original_neighbors):
    total_points = N_x + N_y
    prob_original = hypergeom.pmf(original_neighbors, total_points, N_x, k)
    prob_synthetic = hypergeom.pmf(k - original_neighbors, total_points, N_y, k)
    return prob_original, prob_synthetic

# Compute CDF of Hypergeometric Distribution
def compute_cdf(prob_original):
    cdf = np.cumsum(prob_original)  # This is the cumulative sum of probabilities
    return cdf

# Main execution
original_size = 100
synthetic_size = 100
combined_data, labels = generate_data_with_cluster(original_size=original_size, synthetic_size=synthetic_size)

# Calculate the k-nearest neighbors for each point
k = 10
knn_indices, knn_distances = calculate_knn(combined_data, k)

# Count the original vs synthetic neighbors for each point
counts = count_original_vs_synthetic_neighbors(knn_indices, labels)
#print(counts)
# Get the number of original neighbors for each point
original_neighbors_list = [count[0] for count in counts]
#print(original_neighbors_list)

# Calculate the hypergeometric distribution for each point
N_x = original_size  # Number of original points
N_y = synthetic_size  # Number of synthetic points

# Calculate the probabilities for the original neighbors using hypergeometric distribution
prob_original = np.array([hypergeometric_distribution(N_x,N_y, k, original_neighbors)[0] for original_neighbors in original_neighbors_list])

# Debugging: Print some probabilities to check if they look correct
print("Probabilities for the original neighbors (first 3 values):")
print(prob_original[:3])

# Compute the CDF
cdf = compute_cdf(prob_original)

# Debugging: Check CDF values before plotting
print("CDF values (first 10 values):")
print(cdf[:3])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(cdf, label="CDF of Original Neighbors", color="blue")
plt.xlabel("Point index")
plt.ylabel("CDF")
plt.title("Cumulative Distribution of Original Neighbors in k-NN")
plt.legend()
plt.grid(True)
plt.show()

"""# Display some results for verification
for i in range(10):  # Show the first 5 points' counts and probabilities
    original_count, synthetic_count = counts[i]
    print(f"Point {i+1}: Original neighbors = {original_count}, Synthetic neighbors = {synthetic_count}")
    print(f"Probability of original neighbors: {prob_original[i]}")
    print(f"CDF value: {cdf[i]}")
"""