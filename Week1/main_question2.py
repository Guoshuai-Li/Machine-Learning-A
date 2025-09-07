import numpy as np
import matplotlib.pyplot as plt

def knn(training_points, training_labels, test_points, test_labels):
   """
   - training_points: (d, m) matrix where d=784, m=50
   - training_labels: (m,) array of training labels {-1, 1}
   - test_points: (d, n) matrix where n is test set size
   - test_labels: (n,) array of test labels {-1, 1}
   
   - errors: (m,) array of average errors for K from 1 to m
   """
   d, m = training_points.shape
   d, n = test_points.shape
   
   # Compute distance matrix efficiently using vectorized operations
   # Formula: ||x-z||^2 = x^T*x - 2*x^T*z + z^T*z
   train_norms = np.sum(training_points**2, axis=0)
   test_norms = np.sum(test_points**2, axis=0)
   cross_terms = training_points.T @ test_points
   
   distances = (train_norms[:, np.newaxis] - 2 * cross_terms + 
               test_norms[np.newaxis, :])
   
   # Sort training points by distance for each test point
   sorted_indices = np.argsort(distances, axis=0)
   sorted_labels = training_labels[sorted_indices]
   
   # Compute errors for all K values
   errors = np.zeros(m)
   for k in range(1, m + 1):
       k_nearest_labels = sorted_labels[:k, :]
       predictions = np.sign(np.sum(k_nearest_labels, axis=0))
       predictions[predictions == 0] = 1  # Handle ties
       errors[k-1] = np.mean(predictions != test_labels)
   
   return errors

def create_validation_sets(data_matrix, labels, m, n):
   """
   - data_matrix: full dataset (samples x features)
   - labels: full labels array
   - m: training set size (50)
   - n: validation set size
   
   - X_val_sets: list of 5 validation data matrices (784 x n each)
   - y_val_sets: list of 5 validation label arrays (n each)
   """
   X_val_sets = []
   y_val_sets = []
   
   for i in range(1, 6):  # i from 1 to 5
       start_idx = m + (i - 1) * n  # m + (i-1) * n
       end_idx = m + i * n          # m + i * n
       
       X_val = data_matrix[start_idx:end_idx].T  # shape: (784, n)
       y_val = labels[start_idx:end_idx]         # shape: (n,)
       
       X_val_sets.append(X_val)
       y_val_sets.append(y_val)
   
   return X_val_sets, y_val_sets

if __name__ == "__main__":
   print("Question 2: Digits Classification with K Nearest Neighbors")
   print("="*60)
   
   # Load data files
   data_file_path = "D:/1-LGS/KU course/Machine Learning A/week1/MNIST-5-6-Subset.txt"
   labels_file_path = "D:/1-LGS/KU course/Machine Learning A/week1/MNIST-5-6-Subset-Labels.txt"
   
   # Load image data (reshape to samples x features)
   data_matrix = np.loadtxt(data_file_path).reshape(1877, 784)
   print(f"Data matrix shape: {data_matrix.shape}")
   
   # Load labels and convert from {5,6} to {-1,1} as required
   labels = np.loadtxt(labels_file_path)
   labels = np.where(labels == 5, -1, 1)  # 5 -> -1, 6 -> 1
   print(f"Labels shape: {labels.shape}")
   print(f"Converted labels range: {np.unique(labels)}")
   
   # Set parameters as specified in the task
   m = 50  # number of training samples
   n_values = [10, 20, 40, 80]  # validation set sizes to test
   
   # Extract training data (first m samples)
   X_train = data_matrix[:m].T  # shape: (784, 50) - features x samples
   y_train = labels[:m]         # shape: (50,)
   
   print(f"\nTraining data shape: {X_train.shape}")
   print(f"Training labels shape: {y_train.shape}")
   print(f"Training labels distribution: {np.bincount(y_train + 1)}")  # count of -1 and 1
   
   # Store results for all experiments
   all_results = {}  # Dictionary to store results for each n
   
   # Run experiment for each validation set size
   for n in n_values:
       print(f"\nRunning experiment with n = {n}")
       
       # Create validation sets for current n
       X_val_sets, y_val_sets = create_validation_sets(data_matrix, labels, m, n)
       
       # Store errors for all 5 validation sets
       validation_errors = []
       
       for i in range(5):
           print(f"  Processing validation set {i+1}...")
           errors = knn(X_train, y_train, X_val_sets[i], y_val_sets[i])
           validation_errors.append(errors)
       
       # Convert to numpy array for easier manipulation
       validation_errors = np.array(validation_errors)  # Shape: (5, 50)
       all_results[n] = validation_errors
       
       print(f"  Completed n={n}, result shape: {validation_errors.shape}")
   
   print("\nAll experiments completed!")
   
   # Create plots for each n value (Figure 1)
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.flatten()
   
   K_values = np.arange(1, m + 1)  # K from 1 to 50
   
   for idx, n in enumerate(n_values):
       ax = axes[idx]
       validation_errors = all_results[n]
       
       # Plot each of the 5 validation sets
       for i in range(5):
           ax.plot(K_values, validation_errors[i], 
                   label=f'Validation set {i+1}', alpha=0.7)
       
       ax.set_xlabel('K')
       ax.set_ylabel('Validation Error')
       ax.set_title(f'Validation Error vs K (n = {n})')
       ax.legend()
       ax.grid(True, alpha=0.3)
       ax.set_ylim(0, 1)  # Error rates are between 0 and 1
   
   plt.tight_layout()
   plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
   plt.show()
   
   # Calculate variance across the 5 validation sets for each n and K (Figure 2)
   plt.figure(figsize=(10, 6))
   
   for n in n_values:
       validation_errors = all_results[n]  # Shape: (5, 50)
       
       # Compute variance across the 5 validation sets for each K
       error_variance = np.var(validation_errors, axis=0)  # Shape: (50,)
       
       plt.plot(K_values, error_variance, label=f'n = {n}', marker='o', markersize=3)
   
   plt.xlabel('K')
   plt.ylabel('Variance of Validation Error')
   plt.title('Variance of Validation Error Across 5 Sets vs K')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('figure2.png', dpi=300, bbox_inches='tight')
   plt.show()
   
   # Print analysis results
   print("\nAnalysis Results:")
   print("="*50)
   
   print("\n1. Fluctuations of validation error as a function of n:")
   for n in n_values:
       validation_errors = all_results[n]
       mean_variance = np.mean(np.var(validation_errors, axis=0))
       print(f"   n={n}: Average variance across all K values = {mean_variance:.4f}")
   
   print("\n2. Prediction accuracy as a function of K:")
   for n in n_values:
       validation_errors = all_results[n]
       mean_errors = np.mean(validation_errors, axis=0)
       best_k = np.argmin(mean_errors) + 1
       best_error = mean_errors[best_k-1]
       print(f"   n={n}: Best K = {best_k}, Best average error = {best_error:.3f}")
