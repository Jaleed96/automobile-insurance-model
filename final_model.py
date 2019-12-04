import pandas as pd
import numpy as np
import random
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from joblib import dump, load

training_data = pd.read_csv("./datasets/trainingset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]
train_x = training_data.drop("rowIndex", axis=1, inplace=False)
train_x.drop("ClaimAmount", axis=1, inplace=True)


train_y_categorical = train_y.astype('bool')
train_y_categorical = train_y_categorical.astype('int')



classifier = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1, max_depth=36, bootstrap=False, class_weight="balanced")
classifier.fit(train_x, train_y_categorical)
dump(classifier, "classifier.joblib")

        
regressor = GradientBoostingRegressor(n_estimators=112, random_state=0, max_features=6, max_depth=31, min_samples_leaf=30, learning_rate=0.05)
regressor.fit(train_x, train_y)
dump(regressor, "regressor.joblib")

