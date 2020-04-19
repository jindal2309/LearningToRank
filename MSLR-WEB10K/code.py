
# Sample Code
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file
from LambdaRankNN import LambdaRankNN
from LambdaRankNN import RankNetNN

# generate query data
X = np.array([[0.2, 0.3, 0.4],
              [0.1, 0.7, 0.4],
              [0.3, 0.4, 0.1],
              [0.8, 0.4, 0.3],
              [0.9, 0.35, 0.25]])
y = np.array([0, 1, 0, 0, 2])
qid = np.array([1, 1, 1, 2, 2])

# train model
ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(X, y, qid, epochs=5)
y_pred = ranker.predict(X)
ranker.evaluate(X, y, qid, eval_at=2)


# utils
def sparsity(X):
    number_of_nan = np.count_nonzero(np.isnan(X))
    number_of_zeros = np.count_nonzero(np.abs(X) < 1e-6)
    return (number_of_nan + number_of_zeros) / float(X.shape[0] * X.shape[1]) * 100.

def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return X.todense(), y, queries

def print_dataset_statistics(X, y, queries, name):
    print('----------------------------------')
    print("Characteristics of dataset " + name)
    print("rows x columns " + str(X.shape))
    print("sparsity: " + str(sparsity(X)))
    print("y distribution")
    print(Counter(y))
    print("num samples in queries: minimum, median, maximum")
    print('----------------------------------')

def dump_to_file(out_file_name, X, y, queries):
    all = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))
    pd.DataFrame(all).sort_values(by=[1]).to_csv(out_file_name, sep='\t', header=False, index=False)

def mslr_web(src_path, dst_path):
    train_file = os.path.join(src_path, "train.txt")
    vali_file = os.path.join(src_path, "vali.txt")
    test_file = os.path.join(src_path, "test.txt")

    train_out_file = os.path.join(dst_path, "train.tsv")
    vali_out_file = os.path.join(dst_path, "vali.tsv")
    test_out_file = os.path.join(dst_path, "test.tsv")

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "mslr_web train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(vali_file)
    print_dataset_statistics(X, y, queries, "mslr_web vali")
    dump_to_file(vali_out_file, X, y, queries)
    
    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "mslr_web test")
    dump_to_file(test_out_file, X, y, queries)

# Convert txt to tsv file    
src_path = './Fold1'
dst_path = './Fold1'
mslr_web(src_path, dst_path)


# Read dataset
def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values
    return X, y, queries

train_X, train_y, train_queries = read_dataset(dst_path + "/vali.tsv")
test_X, test_y, test_queries = read_dataset(dst_path + "/test.tsv")
print(train_X.shape, train_y.shape)

train_y = train_y.astype(int)
test_y = test_y.astype(int)

train_X, train_y, train_queries = train_X[:10,:], train_y[:10], train_queries[:10]
test_X, test_y, test_queries =  test_X[:10,:], test_y[:10], test_queries[:10]

# Train LambdaRankNN model
ranker = LambdaRankNN(input_size=train_X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(train_X, train_y, train_queries, epochs=5)
train_y_pred = ranker.predict(train_X)
ranker.evaluate(test_X, test_y, test_queries, eval_at=2)


# Train RankNetNN model
ranker = RankNetNN(input_size=train_X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(train_X, train_y, train_queries, epochs=5)
train_y_pred = ranker.predict(train_X)
ranker.evaluate(test_X, test_y, test_queries, eval_at=2)

