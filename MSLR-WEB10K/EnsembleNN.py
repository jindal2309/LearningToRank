
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
    
y1 = read_data("lambda_mart_max_features_0.8.csv")
y2 = read_data("lambda_mart_max_leaf_nodes_20.csv")
y3 = read_data("lambda_mart_min_samples_leaf_32.csv")
y4 = read_data("lambda_mart_learning_rate_2e-3.csv")
y5 = read_data("lambda_mart_n_estimators_200.csv")
y6 = read_data("lambda_mart.csv")

print(y1.shape, y2.shape, y3.shape)
input_val = torch.cat([y1, y2, y3, y4, y5, y6], dim = 1)

dst_path = './Fold1'
print("Data Loading")
with open(dst_path + '/train.txt') as trainfile, \
        open(dst_path + '/vali.txt') as valifile, \
        open(dst_path + '/test.txt') as evalfile:
    #_, Ty, _, _ = pyltr.data.letor.read_dataset(trainfile)
    #_, Vy, _, _ = pyltr.data.letor.read_dataset(valifile)
    _, Ey, _, _ = pyltr.data.letor.read_dataset(evalfile)

#Ty = Ty[:100000]
#Vy = Vy[:30000]
Ey = Ey[:30000]
print("Data Loading Complete")
targets = torch.tensor(Ey).unsqueeze(1)

num_epochs = 50
batch_size = 50
learning_rate = 1e-2
cuda = True

class SimpleNN1(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, out)
        self.relu = nn.ReLU(inplace = True)
#        nn.init.xavier_uniform_(self.fc1.weight)
#        nn.init.normal_(self.fc1.bias)

    def forward(self, input_val):
        x = self.fc1(input_val)
        out = self.relu(x)
        return out
    
class SimpleNN2(nn.Module):
    def __init__(self, inp, out, hidden_dim ):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out)
        self.relu = nn.ReLU(inplace = True)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias)
        
    def forward(self, input_val):
        x = self.fc1(input_val)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.relu(x)
        return out   
    
    
# Set device type
if cuda and torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print("-" * 84)
print("Running on device type: {}".format(device))

model = SimpleNN1(6,5)
#model = SimpleNN2(3,1,16)
model.to(device)

optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
scheduler = ReduceLROnPlateau(optimizer, "min", patience=100, verbose=True,)

criterion = nn.MSELoss()
training_loss_list = []

model.train()
for epoch in range(num_epochs):
    training_loss = 0
    for i in range(0,len(input_val), batch_size):
        start = i
        end = start + batch_size
        #print(start,end)
        inputs = input_val[start:end,:].float()
        target = targets[start:end].float()
        inputs = inputs.to(device)
        target = target.to(device)
        out = model(inputs)
        #loss = torch.sqrt(criterion(out, target))
        loss = F.nll_loss(out, target)
        training_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        #scheduler.step(training_loss)
    print("epoch", epoch, "training_loss_per_epoch", training_loss)
    training_loss_list.append(training_loss)

print("Input",inputs[:10])
print("Prediction",out[:10])
print("Target", target[:10])
#print(list(model.parameters())[0].data.numpy())