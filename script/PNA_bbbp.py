import sys, os
import tqdm
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import degree

import torch
import torch.nn.functional as F
from torch_geometric.nn import PNAConv

from chem import *

batch_size = 32
dataset_name = 'bbbp'

train_dataset = load_dataset('../data/train_set.csv')
val_dataset = load_dataset('../data/val_set.csv')
test_dataset = load_dataset('../data/test_set.csv')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Compute the maximum in-degree in the training data.
max_degree = -1
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))

# Compute the in-degree histogram tensor
deg = torch.zeros(max_degree + 1, dtype=torch.long)
for data in train_dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.conv1 = PNAConv(25, 16, aggregators=aggregators, scalers=scalers, deg=deg)
        self.conv2 = PNAConv(16, 8, aggregators=aggregators, scalers=scalers, deg=deg)
        self.fc1 = torch.nn.Linear(8,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = global_mean_pool(x, data.batch)
        x = global_add_pool(x, data.batch)
        #x = global_max_pool(x, data.batch)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
        
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in range(10000):
    
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()
    if (epoch+1)%100 == 0:
        print (f"epoch {epoch+1}: ",loss_all / len(train_dataset))
        
        
model.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    pred = model(data).max(dim=1)[1]
    correct += pred.eq(data.y).sum().item()
print ("train set: ", correct / len(train_dataset))

model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    pred = model(data).max(dim=1)[1]
    correct += pred.eq(data.y).sum().item()
print ("test set: ", correct / len(test_dataset))

model.eval()
correct = 0
for data in val_loader:
    data = data.to(device)
    pred = model(data).max(dim=1)[1]
    correct += pred.eq(data.y).sum().item()
print ("val set: ", correct / len(val_dataset))

# Save the model checkpoints 
torch.save(model.state_dict(), 'PNA_bbbp.ckpt')
