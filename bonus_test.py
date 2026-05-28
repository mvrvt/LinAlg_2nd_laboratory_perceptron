import numpy as np
import matplotlib.pyplot as plt
from src.model import Perceptron
from src.utils import standard_scaler

cluster1 = np.random.randn(100, 2) + np.array([2.5, 2.5])
cluster2 = np.random.randn(100, 2) + np.array([-2.5, -2.5])

X = np.vstack([cluster1, cluster2]) 
y = np.array([1] * 100 + [0] * 100)

X_scaled, _ = standard_scaler(X, X)

model = Perceptron(input_dim=2)
model.fit(X_scaled, y, X_scaled, y, epochs=300, lr=0.3, lambda_=0.001)

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='coolwarm', edgecolors='k')

x_vals = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y_vals = -(model.w[0] * x_vals + model.b) / model.w[1]
plt.plot(x_vals, y_vals, 'black', lw=3, label='Идеальная граница')
plt.title(f"Проверка: Accuracy = {np.mean(model.predict(X_scaled) == y):.4f}")
plt.legend()
plt.show()
