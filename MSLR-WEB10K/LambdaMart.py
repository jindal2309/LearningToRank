
import pyltr

with open(dst_path + '/train.txt') as trainfile, \
        open(dst_path + '/vali.txt') as valifile, \
        open(dst_path + '/test.txt') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)
    
metric = pyltr.metrics.NDCG(k=2)

VX1, Vy1, Vqids1 = VX, Vy, Vqids
EX1, Ey1, Eqids1 = EX, Ey, Eqids

TX, Ty, Tqids = VX[:1000], Vy[:1000], Vqids[:1000]
VX, Vy, Vqids = EX[:1000], Ey[:1000], Eqids[:1000]

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
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))

