
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file
from LambdaRankNN import LambdaRankNN

dst_path = './Fold1'
# Read dataset
def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values
    return X, y, queries

train_X, train_y, train_queries = read_dataset(dst_path + "/train.tsv")
test_X, test_y, test_queries = read_dataset(dst_path + "/vali.tsv")
print(train_X.shape, train_y.shape)

train_y = train_y.astype(int)
test_y = test_y.astype(int)

#train_X, train_y, train_queries = train_X[:10,:], train_y[:10], train_queries[:10]
#test_X, test_y, test_queries =  test_X[:10,:], test_y[:10], test_queries[:10]

# Train LambdaRankNN model
ranker = LambdaRankNN(input_size=train_X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(train_X, train_y, train_queries, epochs=5)

test_y_pred = ranker.predict(test_X)
ranker.evaluate(test_X, test_y, test_queries, eval_at=2)

with open('output_lambdarank.txt', 'w') as f:
    f.writelines("%s\n" % i for i in test_y_pred)



