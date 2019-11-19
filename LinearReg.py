import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# Fit linear regression model
# Inputs:
#  X: training input
#  y: training output
# Output:
#  w: estimated weights of the linear regression model
def comp4983_lin_reg_fit(X, y):
    originalX = X
    # add a colume of 1s to X
    X = np.ones((X.shape[0], X.shape[1] + 1))
    X[:, 1:] = originalX
    XT = X.transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), y)
    return w

# Predict using regression model
# Inputs:
#  X: test input
#  w: estimated weights from comp4983_lin_reg_fit()
# Output:
#  y: predicted output
def comp4983_lin_reg_predict(X, w):
    originalX = X
    # add a colume of 1s to X
    X = np.ones((X.shape[0], X.shape[1] + 1))
    X[:, 1:] = originalX
    y = np.dot(X, w)
    return y

data = pd.read_csv("./datasets/trainingset.csv")
data = data.iloc[:, 1:]
print('\n\ndata.info():')
print(data.info())

# split the data into training set and test set
# use 75 percent of the data to train the model and hold back 25 percent
# for testing
train_ratio = 0.75
# number of samples in the data_subset
num_rows = data.shape[0]
# shuffle the indices
shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

# calculate the number of rows for training 
train_set_size = int(num_rows * train_ratio)

# training set: take the first 'train_set_size' rows
train_indices = shuffled_indices[:train_set_size]
# test set: take the remaining rows
test_indices = shuffled_indices[train_set_size:]

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     data_subset = data.iloc[:, [i, data.shape[1] - 1]]
#     train_data = data_subset.iloc[train_indices, :]
#     test_data = data_subset.iloc[test_indices, :]
#     train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#     train_labels = train_data.loc[:, 'ClaimAmount']
#     test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#     test_labels = test_data.loc[:, 'ClaimAmount']
#     w = comp4983_lin_reg_fit(train_features, train_labels)
#     price_pred = comp4983_lin_reg_predict(test_features, w)
#     mae = np.mean(abs(test_labels - price_pred))
#     rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#     total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#     res_sum_sq = sum(pow(test_labels - price_pred, 2))
#     CoD = 1 - (res_sum_sq/total_sum_sq)

#     if mae < best_mae :
#         best_mae = mae
#         best_rmse = rmse
#         best_CoD = CoD
#         best_feature = i

# print('Best feature: ', data.columns[best_feature])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         data_subset = data.iloc[:, [i, j, data.shape[1] - 1]]
#         train_data = data_subset.iloc[train_indices, :]
#         test_data = data_subset.iloc[test_indices, :]
#         train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#         train_labels = train_data.loc[:, 'ClaimAmount']
#         test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#         test_labels = test_data.loc[:, 'ClaimAmount']
#         w = comp4983_lin_reg_fit(train_features, train_labels)
#         price_pred = comp4983_lin_reg_predict(test_features, w)
#         mae = np.mean(abs(test_labels - price_pred))
#         rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#         total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#         res_sum_sq = sum(pow(test_labels - price_pred, 2))
#         CoD = 1 - (res_sum_sq/total_sum_sq)
#         if mae < best_mae :
#             best_mae = mae
#             best_rmse = rmse
#             best_CoD = CoD
#             best_feature_pair_1 = i
#             best_feature_pair_2 = j

# print('Best pair of features: ', data.columns[best_feature_pair_1], data.columns[best_feature_pair_2])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             data_subset = data.iloc[:, [i, j, k, data.shape[1] - 1]]
#             train_data = data_subset.iloc[train_indices, :]
#             test_data = data_subset.iloc[test_indices, :]
#             train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#             train_labels = train_data.loc[:, 'ClaimAmount']
#             test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#             test_labels = test_data.loc[:, 'ClaimAmount']
#             w = comp4983_lin_reg_fit(train_features, train_labels)
#             price_pred = comp4983_lin_reg_predict(test_features, w)
#             mae = np.mean(abs(test_labels - price_pred))
#             rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#             total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#             res_sum_sq = sum(pow(test_labels - price_pred, 2))
#             CoD = 1 - (res_sum_sq/total_sum_sq)

#             if mae < best_mae :
#                 best_mae = mae
#                 best_rmse = rmse
#                 best_CoD = CoD
#                 best_feature_1 = i
#                 best_feature_2 = j
#                 best_feature_3 = k

# print('Best 3 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 data_subset = data.iloc[:, [i, j, k, l, data.shape[1] - 1]]
#                 train_data = data_subset.iloc[train_indices, :]
#                 test_data = data_subset.iloc[test_indices, :]
#                 train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                 train_labels = train_data.loc[:, 'ClaimAmount']
#                 test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                 test_labels = test_data.loc[:, 'ClaimAmount']
#                 w = comp4983_lin_reg_fit(train_features, train_labels)
#                 price_pred = comp4983_lin_reg_predict(test_features, w)
#                 mae = np.mean(abs(test_labels - price_pred))
#                 rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                 total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                 res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                 CoD = 1 - (res_sum_sq/total_sum_sq)

#                 if mae < best_mae :
#                     best_mae = mae
#                     best_rmse = rmse
#                     best_CoD = CoD
#                     best_feature_1 = i
#                     best_feature_2 = j
#                     best_feature_3 = k
#                     best_feature_4 = l

# print('Best 4 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3], data.columns[best_feature_4])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 for m in range(l + 1, data.shape[1] - 1) :
#                     data_subset = data.iloc[:, [i, j, k, l, m, data.shape[1] - 1]]
#                     train_data = data_subset.iloc[train_indices, :]
#                     test_data = data_subset.iloc[test_indices, :]
#                     train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                     train_labels = train_data.loc[:, 'ClaimAmount']
#                     test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                     test_labels = test_data.loc[:, 'ClaimAmount']
#                     w = comp4983_lin_reg_fit(train_features, train_labels)
#                     price_pred = comp4983_lin_reg_predict(test_features, w)
#                     mae = np.mean(abs(test_labels - price_pred))
#                     rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                     total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                     res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                     CoD = 1 - (res_sum_sq/total_sum_sq)

#                     if mae < best_mae :
#                         best_mae = mae
#                         best_rmse = rmse
#                         best_CoD = CoD
#                         best_feature_1 = i
#                         best_feature_2 = j
#                         best_feature_3 = k
#                         best_feature_4 = l
#                         best_feature_5 = m

# print('Best 5 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3], data.columns[best_feature_4], data.columns[best_feature_5])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 for m in range(l + 1, data.shape[1] - 1) :
#                     for n in range(m + 1, data.shape[1] - 1) :
#                         data_subset = data.iloc[:, [i, j, k, l, m, n, data.shape[1] - 1]]
#                         train_data = data_subset.iloc[train_indices, :]
#                         test_data = data_subset.iloc[test_indices, :]
#                         train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                         train_labels = train_data.loc[:, 'ClaimAmount']
#                         test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                         test_labels = test_data.loc[:, 'ClaimAmount']
#                         w = comp4983_lin_reg_fit(train_features, train_labels)
#                         price_pred = comp4983_lin_reg_predict(test_features, w)
#                         mae = np.mean(abs(test_labels - price_pred))
#                         rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                         total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                         res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                         CoD = 1 - (res_sum_sq/total_sum_sq)

#                         if mae < best_mae :
#                             best_mae = mae
#                             best_rmse = rmse
#                             best_CoD = CoD
#                             best_feature_1 = i
#                             best_feature_2 = j
#                             best_feature_3 = k
#                             best_feature_4 = l
#                             best_feature_5 = m
#                             best_feature_6 = n

# print('Best 6 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3], data.columns[best_feature_4], data.columns[best_feature_5], data.columns[best_feature_6])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 for m in range(l + 1, data.shape[1] - 1) :
#                     for n in range(m + 1, data.shape[1] - 1) :
#                         for o in range(n + 1, data.shape[1] - 1) :
#                             data_subset = data.iloc[:, [i, j, k, l, m, n, o, data.shape[1] - 1]]
#                             train_data = data_subset.iloc[train_indices, :]
#                             test_data = data_subset.iloc[test_indices, :]
#                             train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                             train_labels = train_data.loc[:, 'ClaimAmount']
#                             test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                             test_labels = test_data.loc[:, 'ClaimAmount']
#                             w = comp4983_lin_reg_fit(train_features, train_labels)
#                             price_pred = comp4983_lin_reg_predict(test_features, w)
#                             mae = np.mean(abs(test_labels - price_pred))
#                             rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                             total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                             res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                             CoD = 1 - (res_sum_sq/total_sum_sq)

#                             if mae < best_mae :
#                                 best_mae = mae
#                                 best_rmse = rmse
#                                 best_CoD = CoD
#                                 best_feature_1 = i
#                                 best_feature_2 = j
#                                 best_feature_3 = k
#                                 best_feature_4 = l
#                                 best_feature_5 = m
#                                 best_feature_6 = n
#                                 best_feature_7 = o

# print('Best 7 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3], data.columns[best_feature_4], data.columns[best_feature_5], data.columns[best_feature_6], data.columns[best_feature_7])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 for m in range(l + 1, data.shape[1] - 1) :
#                     for n in range(m + 1, data.shape[1] - 1) :
#                         for o in range(n + 1, data.shape[1] - 1) :
#                             for p in range(o + 1, data.shape[1] - 1) :
#                                 data_subset = data.iloc[:, [i, j, k, l, m, n, o, p, data.shape[1] - 1]]
#                                 train_data = data_subset.iloc[train_indices, :]
#                                 test_data = data_subset.iloc[test_indices, :]
#                                 train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                                 train_labels = train_data.loc[:, 'ClaimAmount']
#                                 test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                                 test_labels = test_data.loc[:, 'ClaimAmount']
#                                 w = comp4983_lin_reg_fit(train_features, train_labels)
#                                 price_pred = comp4983_lin_reg_predict(test_features, w)
#                                 mae = np.mean(abs(test_labels - price_pred))
#                                 rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                                 total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                                 res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                                 CoD = 1 - (res_sum_sq/total_sum_sq)

#                                 if mae < best_mae :
#                                     best_mae = mae
#                                     best_rmse = rmse
#                                     best_CoD = CoD
#                                     best_feature_1 = i
#                                     best_feature_2 = j
#                                     best_feature_3 = k
#                                     best_feature_4 = l
#                                     best_feature_5 = m
#                                     best_feature_6 = n
#                                     best_feature_7 = o
#                                     best_feature_8 = p

# print('Best 8 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3], data.columns[best_feature_4], data.columns[best_feature_5], data.columns[best_feature_6], data.columns[best_feature_7], data.columns[best_feature_8])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 for m in range(l + 1, data.shape[1] - 1) :
#                     for n in range(m + 1, data.shape[1] - 1) :
#                         for o in range(n + 1, data.shape[1] - 1) :
#                             for p in range(o + 1, data.shape[1] - 1) :
#                                 for q in range(p + 1, data.shape[1] - 1) :
#                                     data_subset = data.iloc[:, [i, j, k, l, m, n, o, p, q, data.shape[1] - 1]]
#                                     train_data = data_subset.iloc[train_indices, :]
#                                     test_data = data_subset.iloc[test_indices, :]
#                                     train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                                     train_labels = train_data.loc[:, 'ClaimAmount']
#                                     test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                                     test_labels = test_data.loc[:, 'ClaimAmount']
#                                     w = comp4983_lin_reg_fit(train_features, train_labels)
#                                     price_pred = comp4983_lin_reg_predict(test_features, w)
#                                     mae = np.mean(abs(test_labels - price_pred))
#                                     rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                                     total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                                     res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                                     CoD = 1 - (res_sum_sq/total_sum_sq)

#                                     if mae < best_mae :
#                                         best_mae = mae
#                                         best_rmse = rmse
#                                         best_CoD = CoD
#                                         best_feature_1 = i
#                                         best_feature_2 = j
#                                         best_feature_3 = k
#                                         best_feature_4 = l
#                                         best_feature_5 = m
#                                         best_feature_6 = n
#                                         best_feature_7 = o
#                                         best_feature_8 = p
#                                         best_feature_9 = q

# print('Best 9 features: ', data.columns[best_feature_1], data.columns[best_feature_2], data.columns[best_feature_3], data.columns[best_feature_4], data.columns[best_feature_5], data.columns[best_feature_6], data.columns[best_feature_7], data.columns[best_feature_8], data.columns[best_feature_9])
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

# best_mae = float('inf')

# for i in range(data.shape[1] - 1) :
#     for j in range(i + 1, data.shape[1] - 1) :
#         for k in range(j + 1, data.shape[1] - 1) :
#             for l in range(k + 1, data.shape[1] - 1) :
#                 for m in range(l + 1, data.shape[1] - 1) :
#                     for n in range(m + 1, data.shape[1] - 1) :
#                         for o in range(n + 1, data.shape[1] - 1) :
#                             for p in range(o + 1, data.shape[1] - 1) :
#                                 data_subset = data.drop([data.columns[i], data.columns[j], data.columns[k], data.columns[l], data.columns[m], data.columns[n], data.columns[o], data.columns[p]], axis=1)
#                                 train_data = data_subset.iloc[train_indices, :]
#                                 test_data = data_subset.iloc[test_indices, :]
#                                 train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                                 train_labels = train_data.loc[:, 'ClaimAmount']
#                                 test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
#                                 test_labels = test_data.loc[:, 'ClaimAmount']
#                                 w = comp4983_lin_reg_fit(train_features, train_labels)
#                                 price_pred = comp4983_lin_reg_predict(test_features, w)
#                                 mae = np.mean(abs(test_labels - price_pred))
#                                 rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
#                                 total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
#                                 res_sum_sq = sum(pow(test_labels - price_pred, 2))
#                                 CoD = 1 - (res_sum_sq/total_sum_sq)

#                                 if mae < best_mae :
#                                     best_mae = mae
#                                     best_rmse = rmse
#                                     best_CoD = CoD
#                                     dropped_feature_1 = i
#                                     dropped_feature_2 = j
#                                     dropped_feature_3 = k
#                                     dropped_feature_4 = l
#                                     dropped_feature_5 = m
#                                     dropped_feature_6 = n
#                                     dropped_feature_7 = o
#                                     dropped_feature_8 = p

# best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.columns[dropped_feature_3], data.columns[dropped_feature_4], data.columns[dropped_feature_5], data.columns[dropped_feature_6], data.columns[dropped_feature_7], data.columns[dropped_feature_8], data.shape[1] - 1]], axis=1)
# print('Best 10 features: ', best_features.columns)
# print('Mean Absolute Error = ', best_mae)
# print('Root Mean Squared Error = ', best_rmse)
# print('Coefficient of Determination = ', best_CoD)
# print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    for j in range(i + 1, data.shape[1] - 1) :
        for k in range(j + 1, data.shape[1] - 1) :
            for l in range(k + 1, data.shape[1] - 1) :
                for m in range(l + 1, data.shape[1] - 1) :
                    for n in range(m + 1, data.shape[1] - 1) :
                        for o in range(n + 1, data.shape[1] - 1) :
                            data_subset = data.drop([data.columns[i], data.columns[j], data.columns[k], data.columns[l], data.columns[m], data.columns[n], data.columns[o]], axis=1)
                            train_data = data_subset.iloc[train_indices, :]
                            test_data = data_subset.iloc[test_indices, :]
                            train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
                            train_labels = train_data.loc[:, 'ClaimAmount']
                            test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
                            test_labels = test_data.loc[:, 'ClaimAmount']
                            w = comp4983_lin_reg_fit(train_features, train_labels)
                            price_pred = comp4983_lin_reg_predict(test_features, w)
                            mae = np.mean(abs(test_labels - price_pred))
                            rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
                            total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
                            res_sum_sq = sum(pow(test_labels - price_pred, 2))
                            CoD = 1 - (res_sum_sq/total_sum_sq)

                            if mae < best_mae :
                                best_mae = mae
                                best_rmse = rmse
                                best_CoD = CoD
                                dropped_feature_1 = i
                                dropped_feature_2 = j
                                dropped_feature_3 = k
                                dropped_feature_4 = l
                                dropped_feature_5 = m
                                dropped_feature_6 = n
                                dropped_feature_7 = o

best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.columns[dropped_feature_3], data.columns[dropped_feature_4], data.columns[dropped_feature_5], data.columns[dropped_feature_6], data.columns[dropped_feature_7], data.shape[1] - 1], axis=1)
print('Best 11 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    for j in range(i + 1, data.shape[1] - 1) :
        for k in range(j + 1, data.shape[1] - 1) :
            for l in range(k + 1, data.shape[1] - 1) :
                for m in range(l + 1, data.shape[1] - 1) :
                    for n in range(m + 1, data.shape[1] - 1) :
                        data_subset = data.drop([data.columns[i], data.columns[j], data.columns[k], data.columns[l], data.columns[m], data.columns[n]], axis=1)
                        train_data = data_subset.iloc[train_indices, :]
                        test_data = data_subset.iloc[test_indices, :]
                        train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
                        train_labels = train_data.loc[:, 'ClaimAmount']
                        test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
                        test_labels = test_data.loc[:, 'ClaimAmount']
                        w = comp4983_lin_reg_fit(train_features, train_labels)
                        price_pred = comp4983_lin_reg_predict(test_features, w)
                        mae = np.mean(abs(test_labels - price_pred))
                        rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
                        total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
                        res_sum_sq = sum(pow(test_labels - price_pred, 2))
                        CoD = 1 - (res_sum_sq/total_sum_sq)

                        if mae < best_mae :
                            best_mae = mae
                            best_rmse = rmse
                            best_CoD = CoD
                            dropped_feature_1 = i
                            dropped_feature_2 = j
                            dropped_feature_3 = k
                            dropped_feature_4 = l
                            dropped_feature_5 = m
                            dropped_feature_6 = n

best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.columns[dropped_feature_3], data.columns[dropped_feature_4], data.columns[dropped_feature_5], data.columns[dropped_feature_6], data.shape[1] - 1], axis=1)
print('Best 12 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    for j in range(i + 1, data.shape[1] - 1) :
        for k in range(j + 1, data.shape[1] - 1) :
            for l in range(k + 1, data.shape[1] - 1) :
                for m in range(l + 1, data.shape[1] - 1) :
                    data_subset = data.drop([data.columns[i], data.columns[j], data.columns[k], data.columns[l], data.columns[m]], axis=1)
                    train_data = data_subset.iloc[train_indices, :]
                    test_data = data_subset.iloc[test_indices, :]
                    train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
                    train_labels = train_data.loc[:, 'ClaimAmount']
                    test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
                    test_labels = test_data.loc[:, 'ClaimAmount']
                    w = comp4983_lin_reg_fit(train_features, train_labels)
                    price_pred = comp4983_lin_reg_predict(test_features, w)
                    mae = np.mean(abs(test_labels - price_pred))
                    rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
                    total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
                    res_sum_sq = sum(pow(test_labels - price_pred, 2))
                    CoD = 1 - (res_sum_sq/total_sum_sq)

                    if mae < best_mae :
                        best_mae = mae
                        best_rmse = rmse
                        best_CoD = CoD
                        dropped_feature_1 = i
                        dropped_feature_2 = j
                        dropped_feature_3 = k
                        dropped_feature_4 = l
                        dropped_feature_5 = m

best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.columns[dropped_feature_3], data.columns[dropped_feature_4], data.columns[dropped_feature_5], data.shape[1] - 1], axis=1)
print('Best 13 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    for j in range(i + 1, data.shape[1] - 1) :
        for k in range(j + 1, data.shape[1] - 1) :
            for l in range(k + 1, data.shape[1] - 1) :
                data_subset = data.drop([data.columns[i], data.columns[j], data.columns[k], data.columns[l]], axis=1)
                train_data = data_subset.iloc[train_indices, :]
                test_data = data_subset.iloc[test_indices, :]
                train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
                train_labels = train_data.loc[:, 'ClaimAmount']
                test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
                test_labels = test_data.loc[:, 'ClaimAmount']
                w = comp4983_lin_reg_fit(train_features, train_labels)
                price_pred = comp4983_lin_reg_predict(test_features, w)
                mae = np.mean(abs(test_labels - price_pred))
                rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
                total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
                res_sum_sq = sum(pow(test_labels - price_pred, 2))
                CoD = 1 - (res_sum_sq/total_sum_sq)

                if mae < best_mae :
                    best_mae = mae
                    best_rmse = rmse
                    best_CoD = CoD
                    dropped_feature_1 = i
                    dropped_feature_2 = j
                    dropped_feature_3 = k
                    dropped_feature_4 = l

best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.columns[dropped_feature_3], data.columns[dropped_feature_4], data.shape[1] - 1], axis=1)
print('Best 14 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    for j in range(i + 1, data.shape[1] - 1) :
        for k in range(j + 1, data.shape[1] - 1) :
            data_subset = data.drop([data.columns[i], data.columns[j], data.columns[k]], axis=1)
            train_data = data_subset.iloc[train_indices, :]
            test_data = data_subset.iloc[test_indices, :]
            train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
            train_labels = train_data.loc[:, 'ClaimAmount']
            test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
            test_labels = test_data.loc[:, 'ClaimAmount']
            w = comp4983_lin_reg_fit(train_features, train_labels)
            price_pred = comp4983_lin_reg_predict(test_features, w)
            mae = np.mean(abs(test_labels - price_pred))
            rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
            total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
            res_sum_sq = sum(pow(test_labels - price_pred, 2))
            CoD = 1 - (res_sum_sq/total_sum_sq)

            if mae < best_mae :
                best_mae = mae
                best_rmse = rmse
                best_CoD = CoD
                dropped_feature_1 = i
                dropped_feature_2 = j
                dropped_feature_3 = k

best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.columns[dropped_feature_3], data.shape[1] - 1], axis=1)
print('Best 15 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    for j in range(i + 1, data.shape[1] - 1) :
        data_subset = data.drop([data.columns[i], data.columns[j]], axis=1)
        train_data = data_subset.iloc[train_indices, :]
        test_data = data_subset.iloc[test_indices, :]
        train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
        train_labels = train_data.loc[:, 'ClaimAmount']
        test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
        test_labels = test_data.loc[:, 'ClaimAmount']
        w = comp4983_lin_reg_fit(train_features, train_labels)
        price_pred = comp4983_lin_reg_predict(test_features, w)
        mae = np.mean(abs(test_labels - price_pred))
        rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
        total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
        res_sum_sq = sum(pow(test_labels - price_pred, 2))
        CoD = 1 - (res_sum_sq/total_sum_sq)

        if mae < best_mae :
            best_mae = mae
            best_rmse = rmse
            best_CoD = CoD
            dropped_feature_1 = i
            dropped_feature_2 = j

best_features = data.drop([data.columns[dropped_feature_1], data.columns[dropped_feature_2], data.shape[1] - 1], axis=1)
print('Best 16 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

best_mae = float('inf')

for i in range(data.shape[1] - 1) :
    data_subset = data.drop([data.columns[i]], axis=1)
    train_data = data_subset.iloc[train_indices, :]
    test_data = data_subset.iloc[test_indices, :]
    train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
    train_labels = train_data.loc[:, 'ClaimAmount']
    test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
    test_labels = test_data.loc[:, 'ClaimAmount']
    w = comp4983_lin_reg_fit(train_features, train_labels)
    price_pred = comp4983_lin_reg_predict(test_features, w)
    mae = np.mean(abs(test_labels - price_pred))
    rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
    total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
    res_sum_sq = sum(pow(test_labels - price_pred, 2))
    CoD = 1 - (res_sum_sq/total_sum_sq)

    if mae < best_mae :
        best_mae = mae
        best_rmse = rmse
        best_CoD = CoD
        dropped_feature_1 = i

best_features = data.drop([data.columns[dropped_feature_1], data.shape[1] - 1], axis=1)
print('Best 17 features: ', best_features.columns)
print('Mean Absolute Error = ', best_mae)
print('Root Mean Squared Error = ', best_rmse)
print('Coefficient of Determination = ', best_CoD)
print('_')

train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]
train_features = train_data.drop(['ClaimAmount'], axis=1, inplace=False)
train_labels = train_data.loc[:, 'ClaimAmount']
test_features = test_data.drop(['ClaimAmount'], axis=1, inplace=False)
test_labels = test_data.loc[:, 'ClaimAmount']
w = comp4983_lin_reg_fit(train_features, train_labels)
price_pred = comp4983_lin_reg_predict(test_features, w)
mae = np.mean(abs(test_labels - price_pred))
rmse = np.sqrt(np.mean(pow(test_labels - price_pred, 2)))
total_sum_sq = sum(pow(test_labels - np.mean(test_labels), 2))
res_sum_sq = sum(pow(test_labels - price_pred, 2))
CoD = 1 - (res_sum_sq/total_sum_sq)

print('All features')
print('Mean Absolute Error = ', mae)
print('Root Mean Squared Error = ', rmse)
print('Coefficient of Determination = ', CoD)