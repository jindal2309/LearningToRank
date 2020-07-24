import numpy as np
import os
import pandas as pd
from sklearn.datasets import load_svmlight_file


def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return X.todense(), y, queries


def dump_to_file(out_file_name, X, y, queries):
    all = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))
    pd.DataFrame(all).sort_values(by=[1]).to_csv(out_file_name, sep='\t', header=False, index=False)


#Creating tsv files from txt files for Yahoo
def yahoo(src_path, dst_path):

    train_file = os.path.join(src_path, 'set1.train.txt')
    test_file = os.path.join(src_path, 'set1.test.txt')

    train_out_file = os.path.join(dst_path, 'train.tsv')
    test_out_file = os.path.join(dst_path, 'test.tsv')

    X, y, queries = process_libsvm_file(train_file)
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    dump_to_file(test_out_file, X, y, queries)
    

    
def read_dataset(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)

    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 2:].values

    return X, y, queries