from src.data_generator import generate_xor_data, generate_linear_data
from src.model import Perceptron
from src.utils import standard_scaler, get_metrics, cross_validate
import matplotlib.pyplot as plt
import numpy as np

# 1. Сравнение Momentum vs SGD
X, y = generate_linear_data()
X_s, _ = standard_scaler(X, X)

m_mom = Perceptron(2)
m_mom.fit(X_s, y, X_s, y, epochs=100, beta=0.9)

m_sgd = Perceptron(2)
m_sgd.fit(X_s, y, X_s, y, epochs=100, beta=0.0)

plt.figure(figsize=(10, 4))
plt.plot(m_mom.train_loss_history, label='Momentum')
plt.plot(m_sgd.train_loss_history, label='Обычный SGD')
plt.title("Сравнение скорости сходимости")
plt.legend()
plt.show()

# 2. Нелинейные данные: XOR
X_xor, y_xor = generate_xor_data()
X_xor_s, _ = standard_scaler(X_xor, X_xor)
m_xor = Perceptron(2)
m_xor.fit(X_xor_s, y_xor, X_xor_s, y_xor, epochs=200)

plt.scatter(X_xor_s[:, 0], X_xor_s[:, 1], c=m_xor.predict(X_xor_s), cmap='coolwarm')
plt.title("XOR данные: Перцептрон не справляется")
plt.show()

# 3. ROC-Кривая и AUC (Исправлено обращение к словарю)
probs = m_mom.forward(X_s)
metrics = get_metrics(y, m_mom.predict(X_s), probs)

plt.figure()
plt.plot(metrics['roc'][0], metrics['roc'][1], label=f"AUC = {metrics['auc']:.2f}")
plt.plot([0,1],[0,1], 'k--')
plt.title("ROC-кривая")
plt.legend()
plt.show()

# 4. Кросс-валидация
mean, std = cross_validate(X_s, y)
print(f"Кросс-валидация: {mean:.4f} +/- {std:.4f}")
