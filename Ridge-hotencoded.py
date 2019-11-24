import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

def oneHotEncode(data):
    categoricalCol = []
    for i in data:
    #categorical if <= 20 unique values
        if (len(data[i].unique()) <= 20):
            categoricalCol.append(data[i].name)
    data = pd.get_dummies(data, columns = categoricalCol, prefix=categoricalCol)
    return data

# Splitting into x-y
train_y = training_data["ClaimAmount"]

train_no_index = training_data.drop("rowIndex", axis=1, inplace=False)
train_no_index.drop("ClaimAmount", axis=1, inplace=True)
train_x = oneHotEncode(train_no_index)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)
test_x = oneHotEncode(test_x)


ridge_train_errors = []
ridge_cv_errors = []

for i in np.arange(-3, 11):
    lambda_val = math.pow(10, i)
    ridge = Ridge(alpha=lambda_val)

    result = cross_validate(ridge, train_x, train_y, cv=5, scoring='neg_mean_absolute_error', return_train_score=True)
    train_score = result['train_score']
    test_score = result['test_score']

    ridge_train_errors.append(abs(np.sum(train_score) / 5))
    ridge_cv_errors.append(abs(np.sum(test_score) / 5))

plt.plot(np.arange(-3, 11), ridge_train_errors, color="green", label="Training errors")
plt.plot(np.arange(-3, 11), ridge_cv_errors, color="red", label="Validation errors")
plt.xlabel("Lambda values")
plt.ylabel("MAE")
plt.title("5-Fold errors by Lambda value (ridge)")
plt.legend()
plt.show()

ridge = Ridge(alpha=math.pow(10, 4))
ridge.fit(train_x, train_y)

predicted_train_y = ridge.predict(train_x)

ridge_train_mae = np.mean(abs(train_y - predicted_train_y))
print("Training MAE: " + str(ridge_train_mae))

pred_y = ridge.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

weights = ridge.coef_
# absolute since there are negative weights
absolute_weights = np.absolute(weights)
feature_importance_ind = np.argsort(absolute_weights)
print('features least import to most important')
# use sorted indexes to get hot encoded features
print('top 5 features in increasing order', train_x.columns[feature_importance_ind[-5:]])
output.to_csv("./submissions/1_2_6.csv", header=True, index=False)