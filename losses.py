import numpy as np

# Define a function to calculate the mean square error (MSE).
# MSE measures the average squared difference between the true values (y_true) and predicted values (y_pred).


def mse(y_true, y_pred):
    # Convert y_true and y_pred to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        squared_errors = 0
        # raise ValueError("Shapes of y_true and y_pred are not compatible.")
    else:
        # Calculate the squared difference between each corresponding pair of true and predicted values.
        squared_errors = np.power(y_true - y_pred, 2)

    # Calculate the mean of the squared errors, which gives the MSE.
    return np.mean(squared_errors)

# Define a function to calculate the derivative of the mean square error (MSE).
# The derivative is used in gradient descent optimization algorithms for training machine learning models.


def mse_prime(y_true, y_pred):
    # Convert y_true and y_pred to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        y_true = 0  # np.zeros_like(y_pred)
        y_pred = 0  # np.zeros_like(y_true)

    # Calculate the derivative of MSE with respect to y_pred.
    # This derivative is 2 times the difference between y_pred and y_true, divided by the number of data points (np.size(y_true)).
    return 2 * (y_pred - y_true) / np.size(y_true)
