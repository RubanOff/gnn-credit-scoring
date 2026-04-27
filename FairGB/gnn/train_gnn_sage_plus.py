import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

# ------------------- Focal Loss -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss * targets + \
                     (1 - self.alpha) * pt ** self.gamma * BCE_loss * (1 - targets)
        # выравниваем по каждому классу
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
# ------------------------------------------------

# Загрузка категориального графа (лучшего)
data = torch.load('dataset/processed/hcdr_cat_graph.pt')
print(f"Граф загружен: {data}")

# Нормализация
n = data.num_nodes
indices = torch.randperm(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)
train_idx = indices[:train_end]

x = data.x.numpy()
mean = x[train_idx].mean(axis=0, keepdims=True)
std = x[train_idx].std(axis=0, keepdims=True) + 1e-8
data.x = torch.tensor((x - mean) / std, dtype=torch.float)

train_mask = torch.zeros(n, dtype=torch.bool)
val_mask = torch.zeros(n, dtype=torch.bool)
test_mask = torch.zeros(n, dtype=torch.bool)
train_mask[indices[:train_end]] = True
val_mask[indices[train_end:val_end]] = True
test_mask[indices[val_end:]] = True
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Параметры
hidden_dim = 256
num_layers = 3
lr = 0.0003
batch_size = 512           # чуть уменьшен, чтобы не перегружать память
num_neighbors = [10, 5]     # уменьшены из-за огромного числа рёбер
num_epochs = 500
device = torch.device('cpu')
print(f"Device: {device}")
data = data.to(device)

class SAGEPlus(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

model = SAGEPlus(data.num_node_features, hidden_dim, 1, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# Исправленный alpha: вес для положительного класса (дефолтов)
positive_ratio = data.y[train_idx].sum().item() / len(train_idx)
alpha_pos = 0.75  # стандартное значение для редкого класса
criterion = FocalLoss(alpha=alpha_pos, gamma=2)

# DataLoaders
train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                              input_nodes=data.train_mask, shuffle=True)
val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                            input_nodes=data.val_mask, shuffle=False)
test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                             input_nodes=data.test_mask, shuffle=False)

best_val_auc = 0
best_state = None
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].float()
        loss = criterion(out.squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
    scheduler.step()

    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
            val_preds.append(torch.sigmoid(out.squeeze()).cpu())
            val_labels.append(batch.y[:batch.batch_size].cpu())
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        auc = roc_auc_score(val_labels, val_preds)
        f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))
        acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader.dataset):.4f} | Val AUC: {auc:.4f} F1: {f1:.4f} Acc: {acc:.4f}")
        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
model.eval()
test_preds, test_labels = [], []
for batch in test_loader:
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
    test_preds.append(torch.sigmoid(out.squeeze()).cpu())
    test_labels.append(batch.y[:batch.batch_size].cpu())
test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()

test_auc = roc_auc_score(test_labels, test_preds)
test_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
test_f1 = f1_score(test_labels, (test_preds > 0.5).astype(int))
print(f"\nРезультаты SAGE+ с Focal Loss на категориальном графе:")
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test Acc: {test_acc:.4f}")
print(f"  Test F1:  {test_f1:.4f}")