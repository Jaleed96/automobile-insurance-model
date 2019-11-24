import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]
train_x = training_data.drop("rowIndex", axis=1, inplace=False)
train_x.drop("ClaimAmount", axis=1, inplace=True)

# One hot-encoding
# categorical_features = ["feature3", "feature4", "feature5", "feature7", "feature9", "feature11", "feature13", "feature14", "feature15", "feature16", "feature17", "feature18"]
# train_x = pd.get_dummies(train_no_index, columns=categorical_features,  prefix=categorical_features, drop_first=True)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)
# test_x = pd.get_dummies(test_x, columns=categorical_features,  prefix=categorical_features, drop_first=True)

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

print("Validation MAE: " + str(ridge_cv_errors[7]))
print("Training MAE: " + str(ridge_train_mae))

pred_y = ridge.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

output.to_csv("./submissions/1_2_7.csv", header=True, index=False)