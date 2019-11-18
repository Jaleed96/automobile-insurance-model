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

trees_count = [10, 30, 100, 200]

for tree_count in trees_count:
    model = RandomForestRegressor(n_estimators=tree_count, random_state=0)
    model.fit(train_x, train_y)
    train_y_pred = model.predict(train_x)
    cur_mae = metrics.mean_absolute_error(train_y, train_y_pred)
    print("For tree_count = " + str(tree_count) + ", MAE: " + str(cur_mae))

