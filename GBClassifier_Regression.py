import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
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


gb_train_errors = []
gb_cv_errors = []


for i in range(500, 501):
    print(i)
    model = GradientBoostingClassifier(n_estimators=i)
    result = cross_validate(model, train_x, train_y_categorical, cv=6, scoring='f1', return_train_score=True)
    train_score = result['train_score']
    test_score = result['test_score']

    gb_train_errors.append(abs(np.sum(train_score) / 6))
    gb_cv_errors.append(abs(np.sum(test_score) / 6))

print("Training errors: " + str(gb_train_errors))
print("Validation errors: " + str(gb_cv_errors))

plt.plot(np.arange(500, 501), gb_train_errors, color="green", label="Training errors")
plt.plot(np.arange(500, 501), gb_cv_errors, color="red", label="Validation errors")
plt.xlabel("Lambda values")
plt.ylabel("MAE")
plt.title("5-Fold errors by Lambda value (Random Forest)")
plt.legend()
plt.show()

# classifier = RandomForestClassifier(n_estimators=11, random_state=0, max_features=3, n_jobs=-1, max_depth=14, min_samples_leaf=5)
# classifier.fit(train_x, train_y_categorical)

# pred_y_test = classifier.predict(test_x)
# pred_y_train = classifier.predict(train_x)

# claiming_indices = []
# claiming_indices_train = []

# for i in range(len(pred_y_test)):
#     if pred_y_test[i] != 0:
#         claiming_indices.append(i)
        
# for i in range(len(pred_y_train)):
#     if pred_y_train[i] != 0:
#         claiming_indices_train.append(i)

# for i in range(1, 40):
#     model = RandomForestRegressor(n_estimators=30, random_state=0, max_features=3, n_jobs=-1, max_depth=19, min_samples_leaf=i)
#     result = cross_validate(model, train_x_claims_only, train_y_claims_only, cv=6, scoring='neg_mean_absolute_error', return_train_score=True)
#     train_score = result['train_score']
#     test_score = result['test_score']

#     rf_train_errors.append(abs(np.sum(train_score) / 5))
#     rf_cv_errors.append(abs(np.sum(test_score) / 5))

# print("Training error for trees = 30: " + str(rf_train_errors))
# print("Validation error for trees = 30: " + str(rf_cv_errors))

# plt.plot(np.arange(1, 40), rf_train_errors, color="green", label="Training errors")
# plt.plot(np.arange(1, 40), rf_cv_errors, color="red", label="Validation errors")
# plt.xlabel("Lambda values")
# plt.ylabel("MAE")
# plt.title("5-Fold errors by Lambda value (Random Forest)")
# plt.legend()for i in range(1, 40):
#     model = RandomForestRegressor(n_estimators=30, random_state=0, max_features=3, n_jobs=-1, max_depth=19, min_samples_leaf=i)
#     result = cross_validate(model, train_x_claims_only, train_y_claims_only, cv=6, scoring='neg_mean_absolute_error', return_train_score=True)
#     train_score = result['train_score']
#     test_score = result['test_score']

#     rf_train_errors.append(abs(np.sum(train_score) / 5))
#     rf_cv_errors.append(abs(np.sum(test_score) / 5))

# print("Training error for trees = 30: " + str(rf_train_errors))
# print("Validation error for trees = 30: " + str(rf_cv_errors))

# plt.plot(np.arange(1, 40), rf_train_errors, color="green", label="Training errors")
# plt.plot(np.arange(1, 40), rf_cv_errors, color="red", label="Validation errors")
# plt.xlabel("Lambda values")
# plt.ylabel("MAE")
# plt.title("5-Fold errors by Lambda value (Random Forest)")
# plt.legend()
# plt.show()
# plt.show()
        
# regressor = RandomForestRegressor(n_estimators=30, random_state=0, max_features=3, n_jobs=-1, max_depth=19, min_samples_leaf=30)
# regressor.fit(train_x_claims_only, train_y_claims_only)
# for i in range(len(claiming_indices)):
#     cur_sample = np.array(test_x.loc[claiming_indices[i]]).reshape(1, -1)
#     prediction = regressor.predict(cur_sample)
#     pred_y_test[claiming_indices[i]] = prediction

# for i in range(len(claiming_indices_train)):
#     cur_sample = np.array(train_x.loc[claiming_indices_train[i]]).reshape(1, -1)
#     prediction = regressor.predict(cur_sample)
#     pred_y_train[claiming_indices_train[i]] = prediction

# training_mae = np.mean(abs(train_y - pred_y_train))

# print("Training MAE: ", training_mae)

# output = pd.DataFrame({})
# output['rowIndex'] = range(len(pred_y_test))
# output['ClaimAmount'] = pred_y_test

# output.to_csv("./submissions/RFClassifier_Regression.csv", header=True, index=False)