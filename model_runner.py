import pandas as pd
import numpy as np
import random
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from joblib import dump, load

test_data = pd.read_csv("./competitionset.csv")
test_data.drop("rowIndex", axis=1, inplace=True)

classifier = load("classifier.joblib")
regressor = load("regressor.joblib")

y_pred = classifier.predict(test_data)

claim_indices = []

for i in range(len(y_pred)):
    if y_pred[i] != 0:
        claim_indices.append(i)

for i in range(len(claim_indices)):
    cur_sample = np.array(test_data.loc[claim_indices[i]]).reshape(1, -1)
    prediction = regressor.predict(cur_sample)
    print(prediction)
    y_pred[claim_indices[i]] = prediction

output = pd.DataFrame({})
output['rowIndex'] = range(len(y_pred))
output['ClaimAmount'] = y_pred

output.to_csv("./predictedclaimamount.csv", header=True, index=False)