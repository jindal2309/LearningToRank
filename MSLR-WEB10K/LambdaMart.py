'multi:softprob'

import lightgbm as lgb
import pandas as pd
gbm = lgb.LGBMRanker()

dst_path = './Fold1'
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

query_train = [train_X.shape[0]]
#query_val = [X_val.shape[0]]
query_test = [test_X.shape[0]]

gbm.fit(train_X, train_y, group=query_train,
        eval_set=[(test_X, test_y)], eval_group=[query_test],
        eval_at=[5, 10, 20], early_stopping_rounds=50)

test_pred = gbm.predict(test_X)