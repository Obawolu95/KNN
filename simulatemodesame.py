import numpy as np
import matplotlib.pyplot as plt

def generate_data_with_cluster(original_size=100, synthetic_size=100, cluster_ratio=0.05, data_radius=5, cluster_radius=1, min_distance_from_origin=100, random_seed=42):
    """
    Generate original and synthetic datasets, and create a clustered group of synthetic data points
    that are at least 50 units away from the original data.
    
    Parameters:
    - original_size: Number of points in the original dataset (default 100).
    - synthetic_size: Number of points in the synthetic dataset (default 100).
    - cluster_ratio: Fraction of synthetic data to cluster (default 0.05, i.e., 5%).
    - data_radius: Radius to scatter the original and synthetic data points (default 10).
    - cluster_radius: Radius to cluster the outlier points (default 1).
    - min_distance_from_origin: Minimum distance to move the clustered data away from the original data (default 50).
    - random_seed: Random seed for reproducibility (default 42).
    
    Returns:
    - combined_data: Combined dataset of original and synthetic data with the clustered synthetic points.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate original data close to the origin (within a radius of 10)
    original_center = np.array([0, 0])  # Center for original data
    original_data = original_center + np.random.normal(loc=0, scale=data_radius, size=(original_size, 2))
    
    # Generate synthetic data, also close to the original data (radius = 10)
    synthetic_center = np.array([0, 0])  # Center for synthetic data
    synthetic_data = synthetic_center + np.random.normal(loc=0, scale=data_radius, size=(synthetic_size, 2))
    
    # Identify the number of clustered points (5% of synthetic data)
    num_clustered = int(synthetic_size * cluster_ratio)
    
    # Randomly select the indices for the clustered points
    clustered_indices = np.random.choice(synthetic_size, num_clustered, replace=False)
    
    # Move the clustered points far away from the original data (at least 50 units away)
    # First, generate a direction away from the origin (e.g., at a distance of 50)
    angle = np.random.uniform(0, 2 * np.pi, num_clustered)
    radius = np.full(num_clustered, min_distance_from_origin)
    x_offset = radius * np.cos(angle)
    y_offset = radius * np.sin(angle)
    
    # Apply the offset to move the clustered points far from the origin
    synthetic_data[clustered_indices] = synthetic_data[clustered_indices] + np.column_stack((x_offset, y_offset))
    
    # Cluster the selected points within a radius of 1 (moving them slightly to form a small group)
    cluster_center = np.mean(synthetic_data[clustered_indices], axis=0)  # Find the center of the clustered points
    synthetic_data[clustered_indices] = cluster_center + np.random.normal(loc=0, scale=cluster_radius, size=(num_clustered, 2))
    
    # Combine the original data with the synthetic data (including the clustered points)
    combined_data = np.vstack((original_data, synthetic_data))
    
    return combined_data, synthetic_data

# Generate the data with a fixed random seed for reproducibility
combined_data, synthetic_data = generate_data_with_cluster(original_size=100, synthetic_size=100, cluster_ratio=0.01, data_radius=5, cluster_radius=1, min_distance_from_origin=100, random_seed=42)
#print(combined_data)
# Plot the data
plt.figure(figsize=(8, 6))

# Plot original data (blue)
plt.scatter(combined_data[:100, 0], combined_data[:100, 1], color='blue', label='Original Data', alpha=0.7)

# Plot synthetic data (green), including the clustered synthetic points
plt.scatter(combined_data[100:, 0], combined_data[100:, 1], color='green', label='Synthetic Data', alpha=0.7)

# Labels and title
plt.title("Original Data and Synthetic Data with Clustered Points (1%)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
