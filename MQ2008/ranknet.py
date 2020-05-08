
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file
from LambdaRankNN import RankNetNN
from dataloader import load_data

#dst_path = '/bigtemp/ms5sw/MQ2008-agg/Fold1'
dst_path = '/bigtemp/ms5sw/Fold1/'

# Read dataset
train_X, train_y, train_queries = load_data(dst_path + "train.txt")
test_X, test_y, test_queries = load_data(dst_path + "test.txt")
print("Loaded data")
#print(train_X[:5], train_y[:5], train_queries[:5])
print(train_X.shape, train_y.shape)

#train_y = train_y.astype(int)
#test_y = test_y.astype(int)

#train_X, train_y, train_queries = train_X[:100000,:], train_y[:100000], train_queries[:100000]
#test_X, test_y, test_queries =  test_X[:100000,:], test_y[:100000], test_queries[:100000]

# Train RankNetNN model
ranker = RankNetNN(input_size=train_X.shape[1], 
                   hidden_layer_sizes=(16,8,), 
                   activation=('relu', 'relu',), 
                   solver='adam')
epochs = 200
ndcg_at = 10
ranker.fit(train_X, train_y, train_queries, epochs=epochs)

train_y_pred = ranker.predict(train_X)
ranker.evaluate(train_X, train_y, train_queries, eval_at=ndcg_at)
test_y_pred = ranker.predict(test_X)
ranker.evaluate(test_X, test_y, test_queries, eval_at=ndcg_at)

with open('output_ranknet.txt', 'w') as f:
    f.writelines("%s\n" % i for i in test_y_pred)

df1 = pd.DataFrame(train_y_pred)
df3 = pd.DataFrame(test_y_pred)

df = pd.concat([df1, df3], axis = 1)
df.columns = ['Train_pred', 'Test_pred']
df.to_csv("ranknet_sgd.csv", index = False)

