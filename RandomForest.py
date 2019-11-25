import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]
train_no_index = training_data.drop("rowIndex", axis=1, inplace=False)
train_no_index.drop("ClaimAmount", axis=1, inplace=True)

categorical_features = ["feature3", "feature4", "feature5", "feature7", "feature9", "feature11", "feature13", "feature14", "feature15", "feature16", "feature17", "feature18"]
train_x = pd.get_dummies(train_no_index, columns=categorical_features,  prefix=categorical_features, drop_first=True)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)
test_x = pd.get_dummies(test_x, columns=categorical_features,  prefix=categorical_features, drop_first=True)

rf_train_errors = []
rf_cv_errors = []

for i in np.arange(30, 31):
    model = RandomForestRegressor(n_estimators=12, random_state=0, max_features=25, n_jobs=-1, max_depth=31, min_samples_leaf=i)

    result = cross_validate(model, train_x, train_y, cv=5, scoring='neg_mean_absolute_error', return_train_score=True)
    train_score = result['train_score']
    test_score = result['test_score']

    rf_train_errors.append(abs(np.sum(train_score) / 5))
    rf_cv_errors.append(abs(np.sum(test_score) / 5))

plt.plot(np.arange(30, 31), rf_train_errors, color="green", label="Training errors")
plt.plot(np.arange(30, 31), rf_cv_errors, color="red", label="Validation errors")
plt.xlabel("Lambda values")
plt.ylabel("MAE")
plt.title("5-Fold errors by Lambda value (Random Forest)")
plt.legend()
plt.show()


print("Training error for trees = 30: " + str(rf_train_errors))
print("Validation error for trees = 30: " + str(rf_cv_errors))

model = RandomForestRegressor(n_estimators=12, random_state=0, max_features=25, n_jobs=-1, max_depth=31, min_samples_leaf=30)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

output.to_csv("./submissions/1_2_5.csv", header=True, index=False)