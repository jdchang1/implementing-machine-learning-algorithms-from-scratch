import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from math import sqrt
from math import pow

# Parsing CSV file
# Note that I will assume the format is that the last
# column is true value. Furthermore, I will also have
# the assumption that all of my values are numerical
# and that there is no header row (not realistic)
def get_csv(filename):
    data = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        # Clean up blank lines
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    # Note in our hypothesis function, I would like to include theta_0 to simplify prediction
    # thus I added an intercept column in the dataset
    for i in range(len(data)):
        data[i].insert(0, 1.0)
    return data

# Helper method to convert string to float
# Note that I specify the column because some columns
# may include non-numerical values.
def string_to_float(data, column, row = 0):
    for i in range(row, len(data)):
        data[i][column] = float(data[i][column].strip())

# Hypothesis Function or the predictor
def hypothesis(x, theta):
    x = np.mat(x)
    return float(np.asscalar(x.dot(np.transpose(theta))))

# Stochastic Gradient for linear regression
def stoch_gradient(training_data, num_iter, alpha):
    theta = [0.0 for i in range(len(training_data[0])-1)]
    for i in range(num_iter):
        for row in training_data:
            y_hat = hypothesis(row[:-1], theta)
            error = row[-1] - y_hat
            theta[0] = theta[0] + alpha*error
            for j in range(1, len(theta)):
                theta[j] = theta[j] + alpha*error*row[j]
    return theta

# Linear Regression
def linear_regression(data, num_iter, alpha):
    regression = list()
    theta = stoch_gradient(data, num_iter, alpha)
    for row in data:
        y_hat = hypothesis(row[:-1], theta)
        regression.append(y_hat)
    return {"regression": regression, "parameters": theta}

