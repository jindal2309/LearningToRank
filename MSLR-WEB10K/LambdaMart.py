
import warnings
warnings.filterwarnings("ignore")
import pyltr
import pandas as pd
dst_path = './Fold1'

print("Data Loading")
with open(dst_path + '/train.txt') as trainfile, \
        open(dst_path + '/vali.txt') as valifile, \
        open(dst_path + '/test.txt') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

TX, Ty, Tqids = TX[:100000,:], Ty[:100000], Tqids[:100000]
VX, Vy, Vqids =  VX[:30000,:], Vy[:30000], Vqids[:30000]
EX, Ey, Eqids =  EX[:30000,:], Ey[:30000], Eqids[:30000]
print("Data Loading Complete")

print("Train Shape", TX.shape, Ty.shape)
print("Valid Shape", VX.shape, Vy.shape)
print("Test Shape", EX.shape, Ey.shape)

metric = pyltr.metrics.NDCG(k=2)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=200,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1)
print("Model Loaded")

print("Start Training")
model.fit(TX, Ty, Tqids, monitor=monitor)

print("Start Evaluation")
Tpred = model.predict(TX)
Vpred = model.predict(VX)
Epred = model.predict(EX)

print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))

df1 = pd.DataFrame(Tpred)
df2 = pd.DataFrame(Vpred)
df3 = pd.DataFrame(Epred)

df = pd.concat([df1, df2, df3], axis = 1)
df.columns = ['Train_pred', 'Val_pred', 'Test_pred']
df.to_csv("lambda_mart_n_estimators_200.csv", index = False)