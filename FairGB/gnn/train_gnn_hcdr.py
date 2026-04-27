import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

data = torch.load('dataset/processed/hcdr_knn_graph.pt')
print(f"Граф загружен: {data}")

# Нормализация (аналогично)
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

hidden_dim = 128
lr = 0.0005
batch_size = 1024
num_neighbors = [10, 5]
num_epochs = 300
pos_weight = torch.tensor([(len(data.y) - data.y.sum()) / data.y.sum()])

device = torch.device('cpu')
data = data.to(device)

class SAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        return x

model = SAGE(data.num_node_features, hidden_dim, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

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
        y = batch.y[:batch.batch_size]
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), y.float(), pos_weight=pos_weight.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
    scheduler.step()

    if epoch % 20 == 0 or epoch == 1:
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
        acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
        f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader.dataset):.4f} | Val AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
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

print("\nРезультаты SAGE (300 эпох, лучший по val AUC):")
print(f"  Test AUC: {roc_auc_score(test_labels, test_preds):.4f}")
print(f"  Test Acc: {accuracy_score(test_labels, (test_preds > 0.5).astype(int)):.4f}")
print(f"  Test F1:  {f1_score(test_labels, (test_preds > 0.5).astype(int)):.4f}")