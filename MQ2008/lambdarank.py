import numpy as np
import LambdaRankNN
from LambdaRankNN import LambdaRankNN

def sample_data():
    # generate query data
    x = np.array([[0.2, 0.3, 0.4],
                  [0.1, 0.7, 0.4],
                  [0.3, 0.4, 0.1],
                  [0.8, 0.4, 0.3],
                  [0.9, 0.35, 0.25]])
    y = np.array([0, 1, 0, 0, 2])
    q = np.array([1, 1, 1, 2, 2])
    return x, y, q

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        x, y, q = [], [], []
        info = []
        for line in lines:
            # print(line)
            feat, comment = line.split(' #')
            feat = feat.split()
            yi = feat[0]
            qi = feat[1].split(':')[-1]
            xi = feat[2:]
            xi = [f.split(':')[-1] for f in xi]
            # print(yi, qi, xi, comment)
            x.append(xi)
            y.append(yi)
            q.append(qi)
            info.append(comment)
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.int), np.array(q, dtype=np.int)



if __name__ == '__main__':

    X, y, qid = load_data('/Users/mohit/Documents/Grad_Courses/spring20/info-retrieval/learning_to_rank/MQ2008/Fold1/train.txt')
    Xtest, ytest, qid_test = load_data('test.txt')

    epochs = 20
    # train model
    ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu'), solver='adam')
    # ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(32,16,8,), activation=('relu', 'relu', 'relu'), solver='adam')
    ranker.fit(X, y, qid, epochs=epochs)
    y_pred = ranker.predict(Xtest)
    print("Train")
    ranker.evaluate(X, y, qid, eval_at=5)
    ranker.evaluate(X, y, qid, eval_at=10)
    ranker.evaluate(X, y, qid, eval_at=50)
    ranker.evaluate(X, y, qid, eval_at=100)
    print("Test")
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=5)
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=10)
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=50)
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=100)

