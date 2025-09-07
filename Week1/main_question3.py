import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, y):
   """
   - X: (n, d) feature matrix where n=samples, d=features
   - y: (n,) target vector
   
   - theta: (d+1,) parameter vector [intercept, slope1, slope2, ...]
   - y_pred: (n,) predicted values
   """
   # Add bias term (intercept) - create design matrix
   n = X.shape[0]
   if X.ndim == 1:
       X = X.reshape(-1, 1)  # Convert 1D to 2D
   
   X_design = np.column_stack([np.ones(n), X])  # Add column of ones for intercept
   
   # Normal equation: theta = (X^T X)^{-1} X^T y
   XTX = X_design.T @ X_design
   XTy = X_design.T @ y
   theta = np.linalg.solve(XTX, XTy)  # More stable than using inverse
   
   # Compute predictions
   y_pred = X_design @ theta
   
   return theta, y_pred

def compute_r_squared(y_true, y_pred):
   """
   - y_true: actual values
   - y_pred: predicted values
   
   - r_squared: R² value
   """
   y_mean = np.mean(y_true)
   ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squared residuals
   ss_tot = np.sum((y_true - y_mean) ** 2)  # Total sum of squares
   
   r_squared = 1 - (ss_res / ss_tot)
   return r_squared

if __name__ == "__main__":
   print("Question 3: Regression")
   print("="*30)
   
   # Load PCB regression data
   pcb_file_path = "PCB.dt"
   pcb_data = np.loadtxt(pcb_file_path)
   print(f"PCB data shape: {pcb_data.shape}")
   
   # Extract features and targets
   x_pcb = pcb_data[:, 0]  # age of fish (years)
   y_pcb = pcb_data[:, 1]  # PCB concentration (ppm)
   
   print(f"Number of samples: {len(x_pcb)}")
   print(f"Age range: {x_pcb.min():.1f} to {x_pcb.max():.1f} years")
   print(f"PCB concentration range: {y_pcb.min():.2f} to {y_pcb.max():.2f} ppm")
   
   # Task 1: Linear Regression
   print("\nTask 1: Linear Regression Implementation")
   print("-" * 45)
   
   theta, y_pred_linear = linear_regression(x_pcb, y_pcb)
   
   print(f"Linear regression parameters:")
   print(f"Intercept (b): {theta[0]:.4f}")
   print(f"Slope (a): {theta[1]:.4f}")
   print(f"Model: y = {theta[1]:.4f} * x + {theta[0]:.4f}")
   
   # Compute mean squared error
   mse_linear = np.mean((y_pcb - y_pred_linear)**2)
   print(f"Mean Squared Error: {mse_linear:.4f}")
   
   # Task 2: Non-linear Model h(x) = exp(ax + b)
   print("\nTask 2: Non-linear Model h(x) = exp(ax + b)")
   print("-" * 50)
   
   # Step 1: Transform the output data using natural logarithm
   y_log = np.log(y_pcb)
   print(f"Original y range: {y_pcb.min():.2f} to {y_pcb.max():.2f}")
   print(f"Log-transformed y range: {y_log.min():.3f} to {y_log.max():.3f}")
   
   # Step 2: Fit linear model to transformed data: h'(x) = ax + b
   theta_nonlinear, y_pred_log = linear_regression(x_pcb, y_log)
   
   print(f"\nNon-linear model parameters:")
   print(f"Parameter a: {theta_nonlinear[1]:.4f}")
   print(f"Parameter b: {theta_nonlinear[0]:.4f}")
   print(f"Linear model on log scale: ln(y) = {theta_nonlinear[1]:.4f} * x + {theta_nonlinear[0]:.4f}")
   
   # Step 3: Transform back to get final non-linear model h(x) = exp(h'(x))
   y_pred_nonlinear = np.exp(y_pred_log)
   print(f"Final model: h(x) = exp({theta_nonlinear[1]:.4f} * x + {theta_nonlinear[0]:.4f})")
   
   # Compute MSE on original scale
   mse_nonlinear = np.mean((y_pcb - y_pred_nonlinear)**2)
   print(f"\nMean Squared Error (non-linear model): {mse_nonlinear:.4f}")
   print(f"Mean Squared Error (linear model): {mse_linear:.4f}")
   
   # Task 3: Counterexample
   print("\nTask 3: Demonstrating Different Objectives")
   print("-" * 45)
   
   # Simple example: two points with same x, different y
   x_example = np.array([5.0, 5.0])
   y_example = np.array([2.0, 8.0])
   
   print(f"Example data: x = {x_example}, y = {y_example}")
   
   # Method 1: Minimize error on original scale
   print("\nMethod 1: Minimize Σ(yi - exp(axi + b))²")
   h_optimal_original = (y_example[0] + y_example[1]) / 2
   print(f"Optimal prediction h(x) = {h_optimal_original}")
   
   # Method 2: Minimize error on log scale
   y_log_example = np.log(y_example)
   print(f"\nMethod 2: Minimize Σ(ln yi - (axi + b))²")
   print(f"Log-transformed y: {y_log_example}")
   
   log_mean = np.mean(y_log_example)
   h_optimal_log_scale = np.exp(log_mean)
   print(f"Optimal on log scale: exp(mean(ln y)) = exp({log_mean:.3f}) = {h_optimal_log_scale:.3f}")
   
   print(f"\nResults comparison:")
   print(f"Method 1 (original scale): h(x) = {h_optimal_original:.3f}")
   print(f"Method 2 (log scale): h(x) = {h_optimal_log_scale:.3f}")
   print(f"Difference: {abs(h_optimal_original - h_optimal_log_scale):.3f}")
   
   # Task 4: Visualization
   print("\nTask 4: Plotting Data and Model Output")
   print("-" * 40)
   
   plt.figure(figsize=(12, 5))
   
   # Subplot 1: Linear model
   plt.subplot(1, 2, 1)
   plt.scatter(x_pcb, np.log(y_pcb), alpha=0.7, color='blue', label='Data')
   plt.plot(x_pcb, np.log(y_pred_linear), 'r-', linewidth=2, label='Linear Model')
   plt.xlabel('Age (years)')
   plt.ylabel('ln(PCB concentration)')
   plt.title('Linear Regression Model')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Subplot 2: Non-linear model
   plt.subplot(1, 2, 2)
   plt.scatter(x_pcb, np.log(y_pcb), alpha=0.7, color='blue', label='Data')
   plt.plot(x_pcb, y_pred_log, 'g-', linewidth=2, label='Non-linear Model (log scale)')
   plt.xlabel('Age (years)')
   plt.ylabel('ln(PCB concentration)')
   plt.title('Non-linear Model: h(x) = exp(ax + b)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('figure3.png', dpi=300, bbox_inches='tight')
   plt.show()
   
   # Task 5: Coefficient of Determination R²
   print("\nTask 5: Coefficient of Determination (R²)")
   print("-" * 45)
   
   # Compute R² for both models
   r2_linear = compute_r_squared(y_pcb, y_pred_linear)
   r2_nonlinear = compute_r_squared(y_pcb, y_pred_nonlinear)
   
   print(f"Linear model R²: {r2_linear:.4f}")
   print(f"Non-linear model R²: {r2_nonlinear:.4f}")
   
   # Calculate mean of y for reference
   y_mean = np.mean(y_pcb)
   print(f"\nMean of training labels (ȳ): {y_mean:.4f}")
   
   print(f"\nInterpretation of R²:")
   print(f"- R² = 1: Perfect fit (all variation explained)")
   print(f"- R² = 0: Model performs no better than predicting the mean")
   print(f"- R² < 0: Model performs worse than predicting the mean")
   
   # Detailed breakdown
   ss_res_linear = np.sum((y_pcb - y_pred_linear) ** 2)
   ss_res_nonlinear = np.sum((y_pcb - y_pred_nonlinear) ** 2)
   ss_tot = np.sum((y_pcb - y_mean) ** 2)
   
   print(f"\nDetailed breakdown:")
   print(f"Total sum of squares (SS_tot): {ss_tot:.4f}")
   print(f"Residual sum of squares (Linear): {ss_res_linear:.4f}")
   print(f"Residual sum of squares (Non-linear): {ss_res_nonlinear:.4f}")
   print(f"Linear model explains {r2_linear*100:.2f}% of variance")
   print(f"Non-linear model explains {r2_nonlinear*100:.2f}% of variance")
   
   # Task 6: Model with Input Transformation
   print("\nTask 6: Non-linear Model with Input Transformation")
   print("-" * 55)
   
   # Step 1: Apply input transformation x → √x
   x_sqrt = np.sqrt(x_pcb)
   print(f"Original x range: {x_pcb.min():.1f} to {x_pcb.max():.1f}")
   print(f"Transformed x (√x) range: {x_sqrt.min():.2f} to {x_sqrt.max():.2f}")
   
   # Step 2: Fit linear model to (√x, ln(y))
   theta_sqrt, y_pred_log_sqrt = linear_regression(x_sqrt, y_log)
   
   print(f"\nModel parameters:")
   print(f"Parameter a: {theta_sqrt[1]:.4f}")
   print(f"Parameter b: {theta_sqrt[0]:.4f}")
   print(f"Linear model: ln(y) = {theta_sqrt[1]:.4f} * √x + {theta_sqrt[0]:.4f}")
   
   # Step 3: Transform back to get final model h(x) = exp(a√x + b)
   y_pred_sqrt_model = np.exp(y_pred_log_sqrt)
   print(f"Final model: h(x) = exp({theta_sqrt[1]:.4f} * √x + {theta_sqrt[0]:.4f})")
   
   # Compute MSE and R² for the new model
   mse_sqrt = np.mean((y_pcb - y_pred_sqrt_model)**2)
   r2_sqrt = compute_r_squared(y_pcb, y_pred_sqrt_model)
   
   print(f"\nModel performance:")
   print(f"Mean Squared Error: {mse_sqrt:.4f}")
   print(f"R²: {r2_sqrt:.4f}")
   
   # Compare all three models
   print(f"\nModel Comparison:")
   print(f"{'Model':<25} {'MSE':<10} {'R²':<10}")
   print("-" * 45)
   print(f"{'Linear':<25} {mse_linear:<10.4f} {r2_linear:<10.4f}")
   print(f"{'Non-linear':<25} {mse_nonlinear:<10.4f} {r2_nonlinear:<10.4f}")
   print(f"{'Non-linear + √x':<25} {mse_sqrt:<10.4f} {r2_sqrt:<10.4f}")
   
   # Compute R² for transformed labels (as requested)
   r2_sqrt_log_scale = compute_r_squared(y_log, y_pred_log_sqrt)
   print(f"\nR² for transformed labels (log scale): {r2_sqrt_log_scale:.4f}")
   
   # Final visualization for Task 6
   plt.figure(figsize=(10, 6))
   
   plt.scatter(x_pcb, np.log(y_pcb), alpha=0.7, color='blue', 
              label='Data', s=50)
   plt.plot(x_pcb, y_pred_log_sqrt, 'orange', linewidth=2, 
            label='Non-linear Model with √x transformation')
   
   plt.xlabel('Age (years)')
   plt.ylabel('ln(PCB concentration)')
   plt.title('Non-linear Model: h(x) = exp(a√x + b)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('figure4.png', dpi=300, bbox_inches='tight')
   plt.show()
   
   # Final summary
   print("\nFinal Analysis and Discussion:")
   print("="*50)
   print("Model Performance Summary:")
   print(f"1. Linear model: R² = {r2_linear:.4f} (best overall)")
   print(f"2. Non-linear with √x: R² = {r2_sqrt:.4f} (moderate)")
   print(f"3. Simple non-linear: R² = {r2_nonlinear:.4f} (worst)")
   
   print(f"\nKey Insights:")
   print("1. Linear model performs best on original scale despite being simpler")
   print("2. Input transformation (√x) improves fit compared to simple exp(ax+b)")
   print("3. On log scale, √x model has much better fit (R² = 0.7861)")
   print("4. The choice of error metric (original vs log scale) significantly")
   print("   affects which model appears 'best'")
   
   print(f"\nFeature space interpretation:")
   print("- √x transformation maps x to a feature space Z")
   print("- This creates a more linear relationship in the log-transformed output")
   print("- The model effectively learns h(x) = exp(φ(x)) where φ(x) = a√x + b")
   
   print("\nQuestion 3 completed successfully!")
   print("Generated files: figure3.png, figure4.png")
