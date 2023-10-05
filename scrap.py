import numpy as np
import matplotlib.pyplot as plt

# Define the derivative function


def differentiable_f(x):
    return x**2


def derivative_f(x):
    return 2 * x


# Create an array of x values
x = np.linspace(-10, 10, 400)
# print(x)

# Calculate the derivative of x
derivative_values = derivative_f(x)

# Calculate the differentiable of x
differentiable_values = differentiable_f(x)

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the differentiable function
axs[0].plot(x, differentiable_values, label="f(x) = x ^ 2", color='blue')
axs[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axs[0].set_title("Plot of the Differentiable Function")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")
axs[0].legend()
axs[0].grid(True)

# Plot the derivative function
axs[1].plot(x, derivative_values, label="f'(x) = 2x", color='red')
axs[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axs[1].set_title("Plot of the Derivative Function")
axs[1].set_xlabel("x")
axs[1].set_ylabel("f'(x)")
axs[1].legend()
axs[1].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# Plot the differentiable function
plt.figure(figsize=(8, 6))
plt.plot(x, differentiable_values, label="f'(x) = 4x^3 + 3", color='blue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.show()
plt.title("Plot of the Differentiable Function")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()

# Plot the derivative function
plt.figure(figsize=(8, 6))
plt.plot(x, derivative_values, label="f'(x) = 4x^3 + 3", color='blue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.show()
plt.title("Plot of the Derivative Function")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()
