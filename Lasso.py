import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]
train_no_index = training_data.drop("rowIndex", axis=1, inplace=False)
train_no_index.drop("ClaimAmount", axis=1, inplace=True)
train_x = Normalizer().fit_transform(train_no_index)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)
test_x = Normalizer().fit_transform(test_x)

lasso_train_errors = []
lasso_cv_errors = []

for i in np.arange(-4, 2.25, 0.25):
    lambda_val = math.pow(10, i)
    lasso = Lasso(alpha=lambda_val)

    result = cross_validate(lasso, train_x, train_y, cv=5, scoring='neg_mean_absolute_error', return_train_score=True)
    train_score = result['train_score']
    test_score = result['test_score']

    lasso_train_errors.append(abs(np.sum(train_score) / 5))
    lasso_cv_errors.append(abs(np.sum(test_score) / 5))

plt.plot(np.arange(-4, 2.25, 0.25), lasso_train_errors, color="green", label="Training errors")
plt.plot(np.arange(-4, 2.25, 0.25), lasso_cv_errors, color="red", label="Validation errors")
plt.xlabel("Lambda values")
plt.ylabel("MAE")
plt.title("5-Fold errors by Lambda value (Lasso)")
plt.legend()
plt.show()

lasso = Lasso(alpha=math.pow(10, -2))
lasso.fit(train_x, train_y)

predicted_train_y = lasso.predict(train_x)

lasso_train_mae = np.mean(abs(train_y - predicted_train_y))
print("Training MAE: " + str(lasso_train_mae))

pred_y = lasso.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

output.to_csv("./submissions/submission_Lasso.csv", header=True, index=False)