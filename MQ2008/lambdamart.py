
import warnings
warnings.filterwarnings("ignore")
import pyltr
from dataloader import load_data
dst_path = './Fold1'

print("Data Loading")
with open(dst_path + '/train.txt') as trainfile, \
        open(dst_path + '/vali.txt') as valifile, \
        open(dst_path + '/test.txt') as evalfile:
    TX, Ty, Tqids, _ = load_data(trainfile)
    VX, Vy, Vqids, _ = load_data(valifile)
    EX, Ey, Eqids, _ = load_data(evalfile)

print("Data Loading Complete")

metric = pyltr.metrics.NDCG(k=2)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=100,
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
Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))

with open('output_lambdamart.txt', 'w') as f:
    f.writelines("%s\n" % i for i in Epred)
