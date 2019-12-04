import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn import metrics
import matplotlib.pyplot as plt

training_data = pd.read_csv("./datasets/trainingset.csv")

#training_data, test_data = train_test_split(data, train_size=0.75, test_size=0.25, shuffle=False, random_state=42)
test_data = pd.read_csv("./datasets/testset.csv")

# Splitting into x-y
train_y = training_data["ClaimAmount"]
train_x = training_data.drop("rowIndex", axis=1, inplace=False)
train_x.drop("ClaimAmount", axis=1, inplace=True)

test_x = test_data.drop("rowIndex", axis=1, inplace=False)


train_y_categorical = train_y.astype('bool')
train_y_categorical = train_y_categorical.astype('int')

# Adding synthetic samples
sm = SMOTE(random_state=27, ratio=1.0)
tl = TomekLinks(sampling_strategy="majority")
sm_tl = SMOTETomek(random_state=27, smote=sm, tomek=tl)
train_x_synthetic, train_y_categorical_synthetic = sm.fit_sample(train_x, train_y_categorical)

train_x_synthetic = pd.DataFrame(data=train_x_synthetic[0:,0:], columns=test_x.columns)

train_x_claims_only = training_data[training_data.ClaimAmount != 0]
train_x_claims_only.drop("rowIndex", axis=1, inplace=True)
train_x_claims_only.drop("ClaimAmount", axis=1, inplace=True)

train_y_claims_only = train_y[train_y != 0]
print(len(train_y_claims_only), len(train_x) - len(train_x_claims_only))

# rf_train_errors = []
# rf_cv_errors = []


# for i in range(25, 30):
#     model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1, max_depth=28, bootstrap=False, class_weight="balanced")
#     result = cross_validate(model, train_x_synthetic, train_y_categorical_synthetic, cv=6, scoring='f1', return_train_score=True)
#     train_score = result['train_score']
#     test_score = result['test_score']

#     rf_train_errors.append(abs(np.sum(train_score) / 6))
#     rf_cv_errors.append(abs(np.sum(test_score) / 6))
#     print(i)

# print("Training error: " + str(rf_train_errors))
# print("Validation error: " + str(rf_cv_errors))

# plt.plot(np.arange(25, 30), rf_train_errors, color="green", label="Training errors")
# plt.plot(np.arange(25, 30), rf_cv_errors, color="red", label="Validation errors")
# plt.xlabel("Lambda values")
# plt.ylabel("MAE")
# plt.title("5-Fold errors by Lambda value (Random Forest)")
# plt.legend()
# plt.show()

classifier = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1, max_depth=28, bootstrap=False)
classifier.fit(train_x_synthetic, train_y_categorical_synthetic)

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
        
# rf_train_errors = []
# rf_cv_errors = []

# for i in range(1, 5):
#     model = RandomForestRegressor(n_estimators=30, random_state=0, max_features=3, n_jobs=-1, max_depth=19, min_samples_leaf=i)
#     result = cross_validate(model, train_x_claims_only, train_y_claims_only, cv=6, scoring='neg_mean_absolute_error', return_train_score=True,  )
#     train_score = result['train_score']
#     test_score = result['test_score']

#     rf_train_errors.append(abs(np.sum(train_score) / 6))
#     rf_cv_errors.append(abs(np.sum(test_score) / 6))
#     print(i)

# print("Training error for trees = 30: " + str(rf_train_errors))
# print("Validation error for trees = 30: " + str(rf_cv_errors))

# plt.plot(np.arange(1, 5), rf_train_errors, color="green", label="Training errors")
# plt.plot(np.arange(1, 5), rf_cv_errors, color="red", label="Validation errors")
# plt.xlabel("Lambda values")
# plt.ylabel("MAE")
# plt.title("5-Fold errors by Lambda value (Random Forest)")
# plt.legend()
# plt.show()
        
regressor = RandomForestRegressor(n_estimators=30, random_state=0, max_features=3, n_jobs=-1, max_depth=19, min_samples_leaf=30)
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

output.to_csv("./submissions/3_2_9.csv", header=True, index=False)