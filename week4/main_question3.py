import torch
import matplotlib.pyplot as plt
import numpy as np

# Set up matplotlib for LaTeX rendering
usetex = False   
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],  
})

# Define parameters
alpha = 0.5
eta = 0.5  # learning rate
n_iter = 4
r = 1.

# Initialize parameters with requires_grad=True for autograd
# Use no_grad() context to prevent tracking initialization
with torch.no_grad():
    x = torch.tensor([0.9 * r], requires_grad=True)
    y = torch.tensor([0.8 * r], requires_grad=True)

# Define quadratic function
def f(x, y):
    return (alpha * x)**2 + y**2 + alpha * x * y

# Store optimization path
p_x = [x.item()]
p_y = [y.item()]

# Create optimizer
optimizer = torch.optim.SGD([x, y], lr=eta)

# Perform gradient descent optimization
for i in range(n_iter):
    # Reset gradients
    optimizer.zero_grad()
    
    # Compute function value
    loss = f(x, y)
    
    # Backward pass - compute gradients automatically
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Store current position
    p_x.append(x.item())
    p_y.append(y.item())

print("Optimization path:")
for i in range(len(p_x)):
    print(f"Step {i}: x = {p_x[i]:.4f}, y = {p_y[i]:.4f}")

# Create visualization
x_plot = np.linspace(-r, r, 50)
y_plot = np.linspace(-r, r, 50)
X, Y = np.meshgrid(x_plot, y_plot)

# Convert to numpy for plotting
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f(torch.tensor([X[i, j]]), torch.tensor([Y[i, j]])).item()

# Create contour plot
plt.figure(figsize=(8, 6))
contours = plt.contour(X, Y, Z, [0.01, 0.05, 0.1, 0.5, 1.], colors='grey')
plt.clabel(contours, inline=True, fontsize=6)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.imshow(Z, extent=[-r, r, -r, r], origin='lower', cmap='RdGy', alpha=0.5)

# Add optimum
plt.plot(0, 0, 'x', c='k', markersize=10, label='Optimum')

# Plot gradient descent steps
for i in range(n_iter):
    plt.arrow(p_x[i], p_y[i], p_x[i+1]-p_x[i], p_y[i+1]-p_y[i], 
              width=.005, head_width=.045, head_length=.025, 
              length_includes_head=True, fc='b', ec='b', zorder=10)

plt.title('Gradient Descent Optimization with PyTorch Autograd')
plt.legend()
plt.show()
