import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]
train_x = training_data.drop("rowIndex", axis=1, inplace=False)
train_x.drop("ClaimAmount", axis=1, inplace=True)

#categorical_features = ["feature3", "feature4", "feature5", "feature7", "feature9", "feature11", "feature13", "feature14", "feature15", "feature16", "feature17", "feature18"]
#train_x = pd.get_dummies(train_no_index, columns=categorical_features,  prefix=categorical_features, drop_first=True)
train_x_claims_only = training_data[training_data.ClaimAmount != 0]
train_x_claims_only.drop("rowIndex", axis=1, inplace=True)
train_x_claims_only.drop("ClaimAmount", axis=1, inplace=True)

train_y_claims_only = train_y[train_y != 0]

test_x = test_data.drop("rowIndex", axis=1, inplace=False)
#test_x = pd.get_dummies(test_x, columns=categorical_features,  prefix=categorical_features, drop_first=True)

train_y_categorical = train_y.astype('bool')
train_y_categorical = train_y_categorical.astype('int')


rf_train_errors = []
rf_cv_errors = []


# for i in range(2, 100):
#     model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1, max_depth=i, bootstrap=False, class_weight="balanced")
#     result = cross_validate(model, train_x, train_y_categorical, cv=6, scoring='f1', return_train_score=True)
#     train_score = result['train_score']
#     test_score = result['test_score']

#     rf_train_errors.append(abs(np.sum(train_score) / 6))
#     rf_cv_errors.append(abs(np.sum(test_score) / 6))
#     print(i)

# print("Training error: " + str(rf_train_errors))
# print("Validation error: " + str(rf_cv_errors))

# plt.plot(np.arange(2, 100), rf_train_errors, color="green", label="Training f1")
# plt.plot(np.arange(2, 100), rf_cv_errors, color="red", label="Validation f1")
# plt.xlabel("Max-depth")
# plt.ylabel("F1-Score")
# plt.title("6-Fold F1-score by max tree depth (Random Forest)")
# plt.legend()
# plt.show()

classifier = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1, max_depth=36, bootstrap=False, class_weight="balanced")
classifier.fit(train_x, train_y_categorical)

pred_y_test = classifier.predict(test_x)
pred_y_train = classifier.predict(train_x)

claiming_indices = []
claiming_indices_train = []

for i in range(len(pred_y_test)):
    if pred_y_test[i] != 0:
        claiming_indices.append(i)
        
for i in range(len(pred_y_train)):
    if pred_y_train[i] != 0:
        claiming_indices_train.append(i)

for i in range(25, 250, 25):
    model = RandomForestRegressor(n_estimators=i, random_state=0, max_features=2, n_jobs=-1, max_depth=5, min_samples_leaf=40, criterion="mae")
    result = cross_validate(model, train_x, train_y, cv=6, scoring='neg_mean_absolute_error', return_train_score=True)
    train_score = result['train_score']
    test_score = result['test_score']

    rf_train_errors.append(abs(np.sum(train_score) / 6))
    rf_cv_errors.append(abs(np.sum(test_score) / 6))
    print(i)

print("Training error for trees = 30: " + str(rf_train_errors))
print("Validation error for trees = 30: " + str(rf_cv_errors))

plt.plot(np.arange(25, 250, 25), rf_train_errors, color="green", label="Training errors")
plt.plot(np.arange(25, 250, 25), rf_cv_errors, color="red", label="Validation errors")
plt.xlabel("Number of decision trees")
plt.ylabel("MAE")
plt.title("6-Fold errors by n decision trees (Random Forest)")
plt.legend()
plt.show()
        
regressor = RandomForestRegressor(n_estimators=75, random_state=0, max_features=3, n_jobs=-1, max_depth=5, min_samples_leaf=40)
regressor.fit(train_x_claims_only, train_y_claims_only)
for i in range(len(claiming_indices)):
    cur_sample = np.array(test_x.loc[claiming_indices[i]]).reshape(1, -1)
    prediction = regressor.predict(cur_sample)
    pred_y_test[claiming_indices[i]] = prediction

for i in range(len(claiming_indices_train)):
    cur_sample = np.array(train_x.loc[claiming_indices_train[i]]).reshape(1, -1)
    prediction = regressor.predict(cur_sample)
    pred_y_train[claiming_indices_train[i]] = prediction

training_mae = np.mean(abs(train_y - pred_y_train))

print("Training MAE: ", training_mae)

output = pd.DataFrame({})
output['rowIndex'] = range(len(pred_y_test))
output['ClaimAmount'] = pred_y_test

output.to_csv("./submissions/3_2_8.csv", header=True, index=False)