import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fit linear regression model
# Inputs:
#  X: training input
#  y: training output
# Output:
#  w: estimated weights of the linear regression model
def comp4983_lin_reg_fit(X, y):
    originalX = X
    # add a colume of 1s to X
    X = np.ones((X.shape[0], X.shape[1] + 1))
    X[:, 1:] = originalX
    XT = X.transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), y)
    return w

# Predict using regression model
# Inputs:
#  X: test input
#  w: estimated weights from comp4983_lin_reg_fit()
# Output:
#  y: predicted output
def comp4983_lin_reg_predict(X, w):
    originalX = X
    # add a colume of 1s to X
    X = np.ones((X.shape[0], X.shape[1] + 1))
    X[:, 1:] = originalX
    y = np.dot(X, w)
    return y

data = pd.read_csv("./datasets/trainingset.csv")
input_data = data.iloc[:, 1:-1]
output_data = data.iloc[:, -1:]

all_zero = [0] * len(output_data)

mae = np.mean(abs(output_data['ClaimAmount'] - all_zero))

print(mae)