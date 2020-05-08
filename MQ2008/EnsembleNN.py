
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import pandas as pd
import pyltr
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

seed = 142
torch.manual_seed(seed)

def read_data(filename):
    y1 = pd.read_csv(filename)['Test_pred']
    y1 = y1.dropna()
    y1 = torch.tensor(y1).unsqueeze(1)
    return y1

print("Data Loading")  
y1 = read_data("lambda_mart_sgd.csv")
y2 = read_data("lambda_mart_sgd_2.csv")
y3 = read_data("lambda_mart_sgd_3.csv")
y4 = read_data("lambda_mart_sgd_4.csv")
print(y1.shape, y2.shape, y3.shape, y4.shape)
input_val = torch.cat([y1, y2, y3, y4], dim = 1)
print("input_val", input_val.shape)

dst_path = './Fold1'
print("Target Loading")
with open(dst_path + '/train.txt') as trainfile, \
        open(dst_path + '/vali.txt') as valifile, \
        open(dst_path + '/test.txt') as evalfile:
    #_, Ty, _, _ = pyltr.data.letor.read_dataset(trainfile)
    #_, Vy, _, _ = pyltr.data.letor.read_dataset(valifile)
    _, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

#Ty = Ty[:100000]
#Vy = Vy[:30000]
#Ey = Ey[:30000]
#Eqids = Eqids[:30000]
print("Data Loading Complete")
targets = torch.tensor(Ey).unsqueeze(1)

class SimpleNN1(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, out)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
    def forward(self, input_val):
        out = self.fc1(input_val)
        return out
    
class SimpleNN2(nn.Module):
    def __init__(self, inp, out, hdim1, hdim2):
        super().__init__()
        self.fc1 = nn.Linear(inp, hdim1)
        self.fc2 = nn.Linear(hdim1, hdim2)
        self.fc3 = nn.Linear(hdim2, out)
        self.relu = nn.ReLU(inplace = True)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc3.bias)
        
        
    def forward(self, input_val):
        x = self.fc1(input_val)
        x = self.relu(x)
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
# Set device type
if torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")
    
print("-" * 84)
print("Running on device type: {}".format(device))

num_epochs = 2000
batch_size = 30000
learning_rate = 1e-2

#model = SimpleNN1(6,1)
model = SimpleNN2(4,1,8,8)
model.to(device)

#optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
#scheduler = ReduceLROnPlateau(optimizer, "min", patience=100, verbose=True,)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = CyclicLR(optimizer, base_lr=learning_rate, max_lr=1, mode="exp_range", gamma=0.999)

criterion = nn.MSELoss()
training_loss_list = []

model.train()
for epoch in range(num_epochs):
    training_loss = 0
    for i in range(0,len(input_val), batch_size):
        start = i
        end = start + batch_size
        inputs = input_val[start:end,:].float()
        target = targets[start:end].float()
        inputs = inputs.to(device)
        target = target.to(device)
        out = model(inputs)
        optimizer.zero_grad()
        loss = torch.sqrt(criterion(out, target))
        training_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    scheduler.step(training_loss)
    print("epoch", epoch, "training_loss_per_epoch", training_loss)
    training_loss_list.append(training_loss)

#print("Input",inputs[:10])
#print("Prediction",out[:10])
#print("Target", target[:10])

metric = pyltr.metrics.NDCG(k=10)
target1 = target.numpy().reshape(target.shape[0],)
out_r = out.detach().numpy().reshape(out.shape[0],)

print("NDCG Ensemble", metric.calc_mean(Eqids, target1, out_r))

input_mean = input_val.mean(dim=1)
print("NDCG Mean", metric.calc_mean(Eqids, target1, input_mean))

#print(list(model.parameters())[0].data.numpy())
