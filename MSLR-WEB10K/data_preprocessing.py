# Sample Code
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_svmlight_file

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