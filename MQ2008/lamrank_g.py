
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file
from LambdaRankNN import LambdaRankNN
from dataloader import load_data

#dst_path = '/bigtemp/gj3bg/IR/MSLR-Web10K/Fold1'
dst_path = '/bigtemp/gj3bg/IR/MSLR-WEB10K/Fold1'
#dst_path = '/bigtemp/ms5sw/Fold1/'
#dst_path = '/bigtemp/ms5sw/MQ2008-list/Fold1'

# Read dataset
# Read dataset
def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values
    return X, y.astype(int), queries.astype(int)

train_X, train_y, train_queries = read_dataset(dst_path + "/train.tsv")
vali_X, vali_y, vali_queries = read_dataset(dst_path + "/vali.tsv")
test_X, test_y, test_queries = read_dataset(dst_path + "/test.tsv")

#train_y = train_y.astype(int)
#test_y = test_y.astype(int)

train_X, train_y, train_queries = train_X[:100000,:], train_y[:100000], train_queries[:100000]
vali_X, vali_y, vali_queries =  vali_X[:30000,:], vali_y[:30000], vali_queries[:30000]
test_X, test_y, test_queries =  test_X[:30000,:], test_y[:30000], test_queries[:30000]

print("Loaded data")
print(train_X.shape, train_y.shape)

#train_y = train_y.astype(int)
#test_y = test_y.astype(int)

#train_X, train_y, train_queries = train_X[:10,:], train_y[:10], train_queries[:10]
#test_X, test_y, test_queries =  test_X[:10,:], test_y[:10], test_queries[:10]

# Train LambdaRankNN model
ranker = LambdaRankNN(input_size=train_X.shape[1], 
                      hidden_layer_sizes=(16,8,), 
                      activation=('relu', 'relu',), 
                      solver='adam')
epochs = 10
ndcg_at = 10
ranker.fit(train_X, train_y, train_queries, epochs=epochs)

train_y_pred = ranker.predict(train_X)
print("Train evaluate")
ranker.evaluate(train_X, train_y, train_queries, eval_at=ndcg_at)
test_y_pred = ranker.predict(test_X)
print("Test evaluate")
ranker.evaluate(test_X, test_y, test_queries, eval_at=ndcg_at)

with open('output_lambdarank.txt', 'w') as f:
    f.writelines("%s\n" % i for i in test_y_pred)


df1 = pd.DataFrame(train_y_pred)
df3 = pd.DataFrame(test_y_pred)

df = pd.concat([df1, df3], axis = 1)
df.columns = ['Train_pred', 'Test_pred']
df.to_csv("lambda_rank_sgd.csv", index = False)

