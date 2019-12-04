import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

training_data = pd.read_csv("./datasets/trainingset.csv")
test_data = pd.read_csv("./datasets/testset.csv")

# Plotting feature correlations
train_y = training_data["ClaimAmount"]
train_no_index = training_data.drop("rowIndex", axis=1, inplace=False)
train_no_index_standardized = Normalizer().fit_transform(train_no_index)

def plot_two_feats():
    for i in range(len(train_no_index.columns)):
        for j in range(len(train_no_index.columns)):
            if i != j:
                #print(train_no_index_standardized[i:i+1], train_no_index_standardized[j:j+1])
                plt.scatter(train_no_index_standardized[i:i+1], train_no_index_standardized[j:j+1])
                plt.xlabel("Feature " + str(i + 1))
                plt.ylabel("Feature " + str(j + 1))
                plt.title("Feature " + str(i+1) + ' by ' + "Feature " + str(j+1))
                plt.show()

plot_two_feats()