import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.model import Perceptron
from src.utils import standard_scaler, get_metrics

# --- 1. Подготовка данных ---
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_test = standard_scaler(X_train, X_test)

# --- 2. Обучение ---
model = Perceptron(input_dim=2)
model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

# --- 3. Вычисление результатов ---
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("=== Результаты базовой модели ===")
print("Метрики на обучающей выборке:", get_metrics(y_train, train_preds))
print("Метрики на тестовой выборке:", get_metrics(y_test, test_preds))

# --- 4. Визуализация ---
plt.figure(figsize=(12, 5))

# График функции потерь
plt.subplot(1, 2, 1)
plt.plot(model.train_loss_history, label='Обучение')
plt.plot(model.val_loss_history, label='Валидация')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.title('График функции потерь')

# Разделяющая прямая
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
x_vals = np.array([X_test[:, 0].min(), X_test[:, 0].max()])
y_vals = -(model.w[0] * x_vals + model.b) / model.w[1]
plt.plot(x_vals, y_vals, 'black', lw=2, linestyle='--', label='Граница решения')
plt.title('Разделяющая прямая на тесте')
plt.legend()

plt.tight_layout()
plt.show()
