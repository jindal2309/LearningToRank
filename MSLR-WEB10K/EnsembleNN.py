
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR

seed = 142
torch.manual_seed(seed)

y1 = torch.ones(1000,1)
y2 = torch.ones(1000,1)*4
y3 = torch.ones(1000,1)*2

#1000 x 3  
input_val = torch.cat([y1, y2, y3], dim = 1)
targets = torch.ones(1000)*15

num_epochs = 20
batch_size = 10
learning_rate = 1e-3
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

model = SimpleNN1(3,1)
#model = SimpleNN2(3,1,16)
model.to(device)

optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, verbose=True,)

criterion = nn.MSELoss()
training_loss_list = []

model.train()
for epoch in range(num_epochs):
    training_loss = 0
    for i in range(0,len(input_val), batch_size):
        start = i
        end = start + batch_size
        #print(start,end)
        inputs = input_val[start:end,:]
        target = targets[start:end]
        inputs = inputs.to(device)
        out = model(inputs)
        loss = torch.sqrt(criterion(out, target))
        #loss = F.nll_loss(out, target)
        training_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        #scheduler.step(training_loss)
    print("epoch", epoch, "training_loss_per_epoch", training_loss)
    training_loss_list.append(training_loss)

print(out[0:5])
#print(list(model.parameters())[0].data.numpy())