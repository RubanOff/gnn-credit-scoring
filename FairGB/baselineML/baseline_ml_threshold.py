import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

data = torch.load('dataset/processed/hcdr_knn_graph.pt')
X = data.x.numpy()
y = data.y.numpy()

np.random.seed(42)
n = len(y)
indices = np.random.permutation(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)
train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

# Нормализация по train
mean = X[train_idx].mean(axis=0, keepdims=True)
std = X[train_idx].std(axis=0, keepdims=True) + 1e-8
X_norm = (X - mean) / std

# Объединяем train и val для обучения финальной модели, но порог подбираем на val отдельно.
# Подход: обучить на train, подобрать порог на val, потом переобучить на train+val и применить порог к test? 
# Лучше: обучить на train, выбрать порог на val, затем оценить на test. Это честнее, но может быть небольшой объем val.
# Сделаем так: обучим на train, подберём порог на val, протестируем на test. Потом для итогового сравнения приведём обе метрики (с порогом и без).

X_train = X_norm[train_idx]
y_train = y[train_idx]
X_val = X_norm[val_idx]
y_val = y[val_idx]
X_test = X_norm[test_idx]
y_test = y[test_idx]

# Логистическая регрессия
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
y_val_proba = lr.predict_proba(X_val)[:,1]
# Поиск порога
thresholds = np.linspace(0.01, 0.99, 99)
best_f1, best_th = 0, 0.5
for th in thresholds:
    pred = (y_val_proba >= th).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
y_test_proba_lr = lr.predict_proba(X_test)[:,1]
y_test_pred_lr = (y_test_proba_lr >= best_th).astype(int)
print(f"Logistic Regression: best threshold={best_th:.2f}, Test AUC={roc_auc_score(y_test, y_test_proba_lr):.4f}, Acc={accuracy_score(y_test, y_test_pred_lr):.4f}, F1={f1_score(y_test, y_test_pred_lr):.4f}")

# Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_val_proba = rf.predict_proba(X_val)[:,1]
best_f1, best_th = 0, 0.5
for th in thresholds:
    pred = (y_val_proba >= th).astype(int)
    f1 = f1_score(y_val, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th
y_test_proba_rf = rf.predict_proba(X_test)[:,1]
y_test_pred_rf = (y_test_proba_rf >= best_th).astype(int)
print(f"Random Forest: best threshold={best_th:.2f}, Test AUC={roc_auc_score(y_test, y_test_proba_rf):.4f}, Acc={accuracy_score(y_test, y_test_pred_rf):.4f}, F1={f1_score(y_test, y_test_pred_rf):.4f}")