import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from data_utils import get_dataset

# Загружаем Credit Defaulter
data, sens_idx, x_min, x_max = get_dataset('credit')

# Извлекаем признаки и метки
X = data.x.numpy()
y = data.y.numpy()
train_mask = data.train_mask.numpy()
val_mask = data.val_mask.numpy()
test_mask = data.test_mask.numpy()

# Объединяем train и val
train_val_mask = train_mask | val_mask
X_train = X[train_val_mask]
y_train = y[train_val_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Логистическая регрессия
print("Обучение Logistic Regression...")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Случайный лес
print("Обучение Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# Метрики GNN из последнего запуска
gnn_metrics = {
    'Accuracy': 75.43,
    'AUC-ROC': 69.38,
    'F1-score': 84.45
}

# Вывод таблицы
print("\n" + "="*55)
print(f"{'Модель':<22} {'Accuracy':<10} {'AUC-ROC':<10} {'F1-score':<10}")
print("-"*55)
print(f"{'Logistic Regression':<22} {accuracy_score(y_test, y_pred_lr)*100:.2f}%     {roc_auc_score(y_test, y_proba_lr)*100:.2f}%     {f1_score(y_test, y_pred_lr)*100:.2f}%")
print(f"{'Random Forest':<22} {accuracy_score(y_test, y_pred_rf)*100:.2f}%     {roc_auc_score(y_test, y_proba_rf)*100:.2f}%     {f1_score(y_test, y_pred_rf)*100:.2f}%")
print(f"{'GraphSAGE (GNN)':<22} {gnn_metrics['Accuracy']:.2f}%     {gnn_metrics['AUC-ROC']:.2f}%     {gnn_metrics['F1-score']:.2f}%")
print("="*55)

# График сравнения AUC-ROC
models = ['Logistic\nRegression', 'Random\nForest', 'GraphSAGE\n(GNN)']
aucs = [
    roc_auc_score(y_test, y_proba_lr)*100,
    roc_auc_score(y_test, y_proba_rf)*100,
    gnn_metrics['AUC-ROC']
]

plt.figure(figsize=(8,5))
bars = plt.bar(models, aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('AUC-ROC (%)')
plt.title('Сравнение моделей на Credit Defaulter (10 эпох GNN)')
plt.ylim(0, 100)
for bar, val in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.2f}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('model_comparison_credit.png', dpi=150)
plt.show()

print("\nГрафик сохранён как 'model_comparison_credit.png'")