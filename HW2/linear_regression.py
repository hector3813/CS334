"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (N, M+1)
    """
    # TODO: Implement this function
    N = X.shape[0]
    Phi = np.ones((N, M + 1)) 

    for i in range(1, M + 1):
        Phi[:, i] = X[:, 0] ** i
    return Phi

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)
    
    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.
    """
    # TODO: Implement this function
    predictions = np.dot(X, theta)
    loss = np.sum((predictions - y) ** 2) / (2 * X.shape[0])
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    predictions = np.dot(X, theta)
    E_rms = np.sqrt(np.mean((predictions - y) ** 2))
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    theta = np.zeros(d)
    max_iters = int(1e6)  
    tolerance = 1e-10 
    prev_loss = float("inf")

    for _ in range(max_iters):
        predictions = np.dot(X, theta)
        gradient = np.dot(X.T, (predictions - y)) / N
        theta -= learning_rate * gradient

        loss = calculate_squared_loss(X, y, theta)
        if abs(prev_loss - loss) < tolerance:
            break
        prev_loss = loss
    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    theta = np.zeros(d)
    max_iters = int(1e6)  
    min_tolerance = 1e-10  
    prev_loss = float("inf")  
    alpha = 0.001  

    for k in range(max_iters):
        for i in range(N):
            prediction = np.dot(X[i], theta)
            gradient = (prediction - y[i]) * X[i]


            if learning_rate == 'adaptive':
                eta = 0.01 / (1 + alpha * k)  
            else:
                eta = learning_rate

            theta -= eta * gradient
        loss = np.sum((np.dot(X, theta) - y) ** 2) / (2 * N)
        if abs(prev_loss - loss) < min_tolerance:
            return theta  
        prev_loss = loss
    return theta


def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    N, d = X.shape
    I = np.eye(d) 
    

    theta = np.linalg.pinv(X.T @ X + reg_param * I) @ X.T @ y
    
    return theta


''' Uncomment this if you are attempting the extra credit
def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d) 
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    theta = ???
    return theta
'''
def generate_polynomial_features(X, M):
    """
    Generates polynomial features up to degree M.

    Args:
        X: np.array, shape (N, 1) - Input feature
        M: int - Polynomial degree

    Returns:
        Phi: np.array, shape (N, M+1) - Polynomial feature matrix
    """
    N = X.shape[0]
    Phi = np.ones((N, M + 1))
    for i in range(1, M + 1):
        Phi[:, i] = X[:, 0] ** i
    return Phi


def part_1(fname_train):
    """
    Runs gradient descent (GD) and stochastic gradient descent (SGD)
    for different learning rates and prints their results.
    """
    print("========== Part 1 ==========\n")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]

    start = time.process_time()
    theta_adaptive = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate='adaptive')
    elapsed_time = time.process_time() - start

    # TODO: Add more code here to complete part 1
    ##############################
    print("\nGD Results:")
    print("-" * 60)
    print(f"{'η':<10}{'θ₀':<10}{'θ₁':<10}{'# Iterations':<15}{'Runtime (s)':<10}")
    print("-" * 60)

    for eta in learning_rates:
        start = time.process_time()
        theta_gd = ls_gradient_descent(Phi_train, y_train, learning_rate=eta)
        elapsed_time = time.process_time() - start

        print(f"{eta:<10}{theta_gd[0]:<10.6f}{theta_gd[1]:<10.6f}{1000000:<15}{elapsed_time:<10.6f}")

    print("\nSGD Results:")
    print("-" * 60)
    print(f"{'η':<10}{'θ₀':<10}{'θ₁':<10}{'# Iterations':<15}{'Runtime (s)':<10}")
    print("-" * 60)

    for eta in learning_rates:
        start = time.process_time()
        theta_sgd = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=eta)
        elapsed_time = time.process_time() - start

        print(f"{eta:<10}{theta_sgd[0]:<10.6f}{theta_sgd[1]:<10.6f}{1000000:<15}{elapsed_time:<10.6f}")

    print("\nSGD with Adaptive Learning Rate:")
    

    print(f"Adaptive  {theta_adaptive[0]:<10.6f}{theta_adaptive[1]:<10.6f}{1000000:<15}{elapsed_time:<10.6f}")
    
    print("\nDone!")







def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """
    print("=========== Part 2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # TODO: Add more code here to complete part 2
    Phi_train = generate_polynomial_features(X_train, 1)
    Phi_validation = generate_polynomial_features(X_validation, 1)


    theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=0.01)


    train_loss = calculate_squared_loss(Phi_train, y_train, theta)
    validation_loss = calculate_squared_loss(Phi_validation, y_validation, theta)

    print(f"Train Loss: {train_loss:.6f}")
    print(f"Validation Loss: {validation_loss:.6f}")

    print("Done!")



''' Uncomment this if you are attempting the extra credit
def extra_credit(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete extra credit
    """
    print("=========== Extra Credit ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # TODO: Add more code here to complete the extra credit
    ##############################

    print("Done!")
'''


def main(fname_train, fname_validation):
    part_1(fname_train)
    part_2(fname_train, fname_validation)
#    extra_credit(fname_train, fname_validation)


if __name__ == '__main__':
    main("/Users/nimasile/Downloads/HW2_SkeletonCode/data/linreg_train.csv", 
         "/Users/nimasile/Downloads/HW2_SkeletonCode/data/linreg_validation.csv")

