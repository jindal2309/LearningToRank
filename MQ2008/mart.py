import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV

import pandas as pd
import pyltr

if __name__ == '__main__':
    #folder = '/Users/mohit/Documents/Grad_Courses/spring20/info-retrieval/learning_to_rank/MQ2008/Fold1/'
    folder='/bigtemp/ms5sw/Fold1/'
    with open(folder + 'train.txt') as trainfile, \
            open(folder + 'vali.txt') as valifile, \
            open(folder + 'test.txt') as evalfile:
        TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
        VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
        EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

        metric = pyltr.metrics.NDCG(k=10)

        # Only needed if you want to perform validation (early stopping & trimming)
        monitor = pyltr.models.monitors.ValidationMonitor(
            VX, Vy, Vqids, metric=metric, stop_after=100)
        """
        mart = pyltr.models.LambdaMART(
            metric=metric, 
            query_subsample=0.5,
            max_leaf_nodes=10,
            min_samples_leaf=64,
            verbose=1)
        params = {'n_estimators':[500, 1000, 2000, 5000], 'learning_rate':[0.01, 0.02, 0.1, 0.2, 0.5], 'subsample':[0.5,1], 'max_features':[0.5,'auto']}
        grid = GridSearchCV(mart, params)
        grid.fit(TX, Ty, Tqids, monitor=monitor)
        print("Grid search results")
        print(grid.cv_results_.keys())
        """
        ranker = pyltr.models.LambdaMART(
            metric=metric,
            n_estimators=1000,
            learning_rate=0.01,
            max_features=0.5,
            query_subsample=0.5,
            max_leaf_nodes=10,
            min_samples_leaf=16,
            verbose=1,
        )

        ranker.fit(TX, Ty, Tqids, monitor=monitor)
        #model.fit(TX, Ty, Tqids)

        train_y_pred = ranker.predict(TX)
        #ranker.evaluate(TX, Ty, Tqids, eval_at=10)
        print('Train NDCG@10:', metric.calc_mean(Tqids, Ty, train_y_pred))
        
        test_y_pred = ranker.predict(EX)
        #ranker.evaluate(EX, Ey, Eqids, eval_at=10)
        print('Test NDCG@10:', metric.calc_mean(Eqids, Ey, test_y_pred))

        with open('output_lambdamart.txt', 'w') as f:
            f.writelines("%s\n" % i for i in test_y_pred)

        df1 = pd.DataFrame(train_y_pred)
        df3 = pd.DataFrame(test_y_pred)
        df = pd.concat([df1, df3], axis = 1)
        df.columns = ['Train_pred',  'Test_pred']
        df.to_csv("lambda_mart_sgd_4.csv", index = False)
        
        
