import numpy as np
import matplotlib.pyplot as plt

def generate_data_within_radius(original_size=100, synthetic_size=100, original_radius=2, synthetic_radius=20, random_seed=42):
    """
    Generate synthetic data within a larger radius and original data within a smaller radius from the origin.
    
    Parameters:
    - original_size: Number of points in the original dataset (default 100).
    - synthetic_size: Number of points in the synthetic dataset (default 100).
    - original_radius: Radius of the original data (default 2).
    - synthetic_radius: Radius of the synthetic data (default 20).
    - random_seed: Random seed for reproducibility (default 42).
    
    Returns:
    - original_data: Original dataset within the specified radius.
    - synthetic_data: Synthetic dataset within a smaller radius.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate synthetic data points within a radius of 'synthetic_radius' (radius 20)
    synthetic_center = np.array([0, 0])  # Center for synthetic data
    synthetic_data = synthetic_center + np.random.normal(loc=0, scale=synthetic_radius, size=(synthetic_size, 2))
    
    # Generate original data points within a radius of 'original_radius' (radius 2)
    original_center = np.array([0, 0])  # Center for original data
    original_data = original_center + np.random.normal(loc=0, scale=original_radius, size=(original_size, 2))
    
    return original_data, synthetic_data

# Generate the data with swapped roles
original_data, synthetic_data = generate_data_within_radius(original_size=100, synthetic_size=100, original_radius=2, synthetic_radius=20, random_seed=42)

# Plot the data
plt.figure(figsize=(8, 6))

# Plot original data (green) which now has radius 2
plt.scatter(original_data[:, 0], original_data[:, 1], color='green', label='Original Data', alpha=0.7)

# Plot synthetic data (blue) which now has radius 20
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], color='blue', label='Synthetic Data', alpha=0.7)

# Labels and title
plt.title("Swapped: Synthetic Data (Blue) and Original Data (Green) with Different Radii")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
