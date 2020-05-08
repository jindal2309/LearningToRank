
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file
from LambdaRankNN import RankNetNN

dst_path = './Fold1'

# Read dataset
def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values
    return X, y, queries

train_X, train_y, train_queries = read_dataset(dst_path + "/train.tsv")
vali_X, vali_y, vali_queries = read_dataset(dst_path + "/vali.tsv")
test_X, test_y, test_queries = read_dataset(dst_path + "/vali.tsv")

train_y = train_y.astype(int)
test_y = test_y.astype(int)

train_X, train_y, train_queries = train_X[:100000,:], train_y[:100000], train_queries[:100000]
vali_X, vali_y, vali_queries =  vali_X[:30000,:], vali_y[:30000], vali_queries[:30000]
test_X, test_y, test_queries =  test_X[:30000,:], test_y[:30000], test_queries[:30000]

print("Train Shape", train_X.shape, train_y.shape)
print("Valid Shape", vali_X.shape, vali_y.shape)
print("Test Shape", test_X.shape, test_y.shape)

# Train RankNetNN model
ranker = RankNetNN(input_size=train_X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(train_X, train_y, train_queries, epochs=10)

train_y_pred = ranker.predict(train_X)
vali_y_pred = ranker.predict(vali_X)
test_y_pred = ranker.predict(test_X)
ranker.evaluate(test_X, test_y, test_queries, eval_at=2)

df1 = pd.DataFrame(train_y_pred)
df2 = pd.DataFrame(vali_y_pred)
df3 = pd.DataFrame(test_y_pred)

df = pd.concat([df1, df2, df3], axis = 1)
df.columns = ['Train_pred', 'Val_pred', 'Test_pred']
df.to_csv("rank_net.csv", index = False)
