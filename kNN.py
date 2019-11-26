import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("./datasets/trainingset.csv")
data = data.drop("rowIndex", axis=1, inplace=False)

train_ratio = 0.75
num_rows = data.shape[0]
train_set_size = int(num_rows * train_ratio)
train_data = data.iloc[:train_set_size, :]
test_data = data.iloc[train_set_size:, :]
print(len(train_data), "training samples + ", len(test_data), "test samples")
train_output = train_data["ClaimAmount"]
train_input = train_data.drop("ClaimAmount", axis=1, inplace=False)
test_output = test_data["ClaimAmount"]
test_input = test_data.drop("ClaimAmount", axis=1, inplace=False)

trainY = []
testY = []

for i in train_output:
    trainY.append(1 if i > 0 else 0)

for i in test_output:
    testY.append(1 if i > 0 else 0)

best_gain = -100
best_k = 1
zero_in_test = len(testY) - sum(testY)

for k in range(1, 25):
    print("Evaluating K = ", k)
    knn = KNeighborsClassifier(k)
    knn.fit(train_input, trainY)
    test_pred = knn.predict(test_input)
    correct_results = sum(test_pred == testY)
    gain = correct_results - zero_in_test
    if gain > best_gain:
        best_gain = gain
        best_k = k

knn = KNeighborsClassifier(best_k)

knn.fit(train_input, trainY)
test_pred = knn.predict(test_input)

correct1 = 0
correct0 = 0

for i in range(len(test_pred)):
    if test_pred[i] == testY[i]:
        if test_pred[i] == 0:
            correct0 += 1
        else:
            correct1 += 1

print("Best K = ", best_k)
print("Number of 0's in test set: ", zero_in_test)
print("Number of 1's in test set: ", sum(testY))
print("Number of correct predictions of 0: {} | {} ({}%)".format(correct0, zero_in_test, correct0/zero_in_test*100))
print("Number of correct predictions of 1: {} | {} ({}%)".format(correct1, sum(testY), correct1/sum(testY)*100))
print("Number of correct predictions: {} | {} ({}%)".format(correct0 + correct1, len(testY), (correct0 + correct1)/len(testY)*100))
print("All zero prediction: {} | {} ({}%)".format(zero_in_test, len(testY), zero_in_test/len(testY)*100))
print("Gain over all zero by using KNN: {}%".format((correct0 + correct1)/len(testY)*100 - zero_in_test/len(testY)*100))