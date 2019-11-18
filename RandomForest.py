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



model = RandomForestRegressor(n_estimators=50, random_state=0)
model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)
cur_mae = metrics.mean_absolute_error(train_y, train_y_pred)
print("For tree_count = " + str(50) + ", training MAE: " + str(cur_mae))

pred_y = model.predict(test_x)
output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y))
output['claimAmount'] = pred_y

output.to_csv("./submissions/submission_RandomForest.csv", header=True, index=False)