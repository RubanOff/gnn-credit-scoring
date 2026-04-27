import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Загружаем граф
data = torch.load('dataset/processed/hcdr_knn_graph.pt')
X = data.x.numpy()
y = data.y.numpy()

# Разбиение (то же, что и в GNN, для честного сравнения)
np.random.seed(42)
n = len(y)
indices = np.random.permutation(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# НОРМАЛИЗАЦИЯ: используем статистики ТОЛЬКО из train
mean = X[train_idx].mean(axis=0, keepdims=True)
std = X[train_idx].std(axis=0, keepdims=True) + 1e-8
X_norm = (X - mean) / std

# Объединяем train и val для обучения (как в предыдущих бейзлайнах)
train_val_idx = np.concatenate([train_idx, val_idx])
X_train_val = X_norm[train_val_idx]
y_train_val = y[train_val_idx]
X_test = X_norm[test_idx]
y_test = y[test_idx]

# Логистическая регрессия
print("Обучение Logistic Regression...")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_val, y_train_val)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Случайный лес
print("Обучение Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_val, y_train_val)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Метрики GNN (из последнего запуска)
gnn_auc = 0.7130
gnn_acc = 0.6689
gnn_f1 = 0.2348

print("\n" + "="*60)
print(f"{'Модель':<22} {'AUC-ROC':<10} {'Accuracy':<10} {'F1-score':<10}")
print("-"*60)
print(f"{'Logistic Regression':<22} {roc_auc_score(y_test, y_proba_lr)*100:.2f}%      {accuracy_score(y_test, y_pred_lr)*100:.2f}%      {f1_score(y_test, y_pred_lr)*100:.2f}%")
print(f"{'Random Forest':<22} {roc_auc_score(y_test, y_proba_rf)*100:.2f}%      {accuracy_score(y_test, y_pred_rf)*100:.2f}%      {f1_score(y_test, y_pred_rf)*100:.2f}%")
print(f"{'GraphSAGE (GNN)':<22} {gnn_auc*100:.2f}%      {gnn_acc*100:.2f}%      {gnn_f1*100:.2f}%")
print("="*60)