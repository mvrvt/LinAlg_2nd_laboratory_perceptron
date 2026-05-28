from src.data_generator import generate_linear_data, generate_circles_data
from src.model import Perceptron
from src.utils import standard_scaler, get_metrics, cross_validate
import matplotlib.pyplot as plt
import numpy as np

# 1. Тест на нелинейных данных (Круги)
X, y = generate_circles_data(n_samples=600)
X_scaled, _ = standard_scaler(X, X)

model = Perceptron(input_dim=2)
model.fit(X_scaled, y, X_scaled, y, epochs=200, lr=0.1)

# Визуализация ошибок (Задание 3)
preds = model.predict(X_scaled)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[preds == y, 0], X_scaled[preds == y, 1], c='green', label='Правильно', alpha=0.3)
plt.scatter(X_scaled[preds != y, 0], X_scaled[preds != y, 1], c='red', marker='x', label='Ошибка')
plt.title("Классификация кругов (Перцептрон бессилен)")
plt.legend()

# 2. ROC-кривая (Задание 3)
X_lin, y_lin = generate_linear_data(noise=0.05)
X_l_scaled, _ = standard_scaler(X_lin, X_lin)
model_lin = Perceptron(2)
model_lin.fit(X_l_scaled, y_lin, X_l_scaled, y_lin, epochs=100, lr=0.1)

probs = model_lin.forward(X_l_scaled)
metrics = get_metrics(y_lin, model_lin.predict(X_l_scaled), probs)

plt.subplot(1, 2, 2)
plt.plot(metrics['roc'][0], metrics['roc'][1], label=f"AUC = {metrics['AUC']:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-кривая")
plt.legend()
plt.show()

# 3. Кросс-валидация (Задание 5)
mean_acc, std_acc = cross_validate(X_l_scaled, y_lin)
print(f"Результат К-fold: {mean_acc:.4f} (+/- {std_acc:.4f})")
