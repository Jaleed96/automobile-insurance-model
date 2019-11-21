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
train_x = Normalizer().fit_transform(train_no_index)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)
test_x = Normalizer().fit_transform(test_x)

train_y_categorical = train_y.astype('bool')


rf_train_errors = -1
rf_cv_errors = -1


model = RandomForestRegressor(n_estimators=30, random_state=0)

result = cross_validate(model, train_x, train_y_categorical, cv=5, scoring='neg_mean_absolute_error', return_train_score=True)
train_score = result['train_score']
test_score = result['test_score']


rf_train_errors = abs(np.mean(train_score))
rf_cv_errors = abs(np.mean(test_score))

print("Training error for trees = 30: " + str(rf_train_errors))
print("Validation error for trees = 30: " + str(rf_cv_errors))

model = RandomForestRegressor(n_estimators=30, random_state=0)
model.fit(train_x, train_y_categorical)

pred_y = model.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

output.to_csv("./submissions/submission_RandomForest_Regression.csv", header=True, index=False)