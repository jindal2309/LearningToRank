# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:04:15 2020

@author: Rishab
"""

import numpy as np
from LambdaRankNN import LambdaRankNN
import os
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file

#X1 = np.array([[0.2, 0.3, 0.4],
#              [0.1, 0.7, 0.4],
#              [0.3, 0.4, 0.1],
#              [0.8, 0.4, 0.3],
#              [0.9, 0.35, 0.25]])
#y1 = np.array([0, 1, 0, 0, 2])
#qid = np.array([1, 1, 1, 2, 2])
#
#ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
#ranker.fit(X, y, qid, epochs=5)
#y_pred = ranker.predict(X)
#ranker.evaluate(X, y, qid, eval_at=2)

def sparsity(X):
    number_of_nan = np.count_nonzero(np.isnan(X))
    number_of_zeros = np.count_nonzero(np.abs(X) < 1e-6)
    return (number_of_nan + number_of_zeros) / float(X.shape[0] * X.shape[1]) * 100.


def print_dataset_statistics(X, y, queries, name):
    print('----------------------------------')
    print("Characteristics of dataset " + name)
    print("rows x columns " + str(X.shape))
    print("sparsity: " + str(sparsity(X)))
    print("y distribution")
    print(Counter(y))
    print("num samples in queries: minimum, median, maximum")
    #num_queries = Counter(queries).values()
    #print(np.min(num_queries), np.median(num_queries), np.max(num_queries))
    print('----------------------------------')


def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return X.todense(), y, queries


def dump_to_file(out_file_name, X, y, queries):
    all = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))
    pd.DataFrame(all).sort_values(by=[1]).to_csv(out_file_name, sep='\t', header=False, index=False)


def yahoo(src_path, dst_path):
    """
    0 - label, 1 - qid, ...features...
    ----------------------------------
    Characteristics of dataset yahoo train
    rows x columns (473134, 699)
    sparsity: 68.1320434932
    y distribution
    Counter({1.0: 169897, 2.0: 134832, 0.0: 123294, 3.0: 36170, 4.0: 8941})
    num samples in queries: minimum, median, maximum
    (1, '19.0', 139)
    ----------------------------------
    ----------------------------------
    Characteristics of dataset yahoo test
    rows x columns (165660, 699)
    sparsity: 68.0674251017
    y distribution
    Counter({1.0: 59107, 2.0: 48033, 0.0: 42625, 3.0: 12804, 4.0: 3091})
    num samples in queries: minimum, median, maximum
    (1, '19.0', 129)
    ----------------------------------
    """

    train_file = os.path.join(src_path, 'set1.train.txt')
    test_file = os.path.join(src_path, 'set1.test.txt')

    train_out_file = os.path.join(dst_path, 'train.tsv')
    test_out_file = os.path.join(dst_path, 'test.tsv')

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "yahoo train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "yahoo test")
    dump_to_file(test_out_file, X, y, queries)
    
#yahoo('./data', './data_output')
    
def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)

    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values

    # assert np.all(queries == np.sort(queries))

    return X, y, queries


X, y, queries = read_dataset('./data_output/train.tsv')
y = y.astype(int)
queries = queries.astype(int)
ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(X, y, queries, epochs=5)
y_pred = ranker.predict(X)
ranker.evaluate(X, y, queries, eval_at=2)
    



