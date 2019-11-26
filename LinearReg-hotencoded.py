import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

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

def oneHotEncode(data):
    categoricalCol = []
    for i in data:
    #categorical if <= 20 unique values
        if (len(data[i].unique()) <= 20):
            categoricalCol.append(data[i].name)
    data = pd.get_dummies(data, columns = categoricalCol, prefix=categoricalCol)
    return data

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]

train_no_index = training_data.drop("rowIndex", axis=1, inplace=False)
train_no_index.drop("ClaimAmount", axis=1, inplace=True)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)

train_x = train_no_index.loc[:, ['feature1','feature2','feature4','feature6','feature7','feature10','feature11','feature16','feature18']]
train_x = oneHotEncode(train_x)
test_x = test_x.loc[:, ['feature1','feature2','feature4','feature6','feature7','feature10','feature11','feature16','feature18']]
test_x = oneHotEncode(test_x)
lin_reg = LinearRegression()
lin_reg.fit(train_x, train_y)
pred_train_y = lin_reg.predict(train_x)
mae = np.mean(abs(train_y - pred_train_y))
print('Mean Absolute Error = ', mae)

pred_y = lin_reg.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

output.to_csv("./submissions/2_2_3.csv", header=True, index=False)