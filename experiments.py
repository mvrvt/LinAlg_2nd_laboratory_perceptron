# experiments.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.model import Perceptron
from src.utils import standard_scaler, get_metrics

# 1. Подготовка данных (такая же, как в main)
X, y = make_classification( n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1 )
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, stratify=y, random_state=42 )
X_train, X_test = standard_scaler( X_train, X_test )

# 2. Список параметров для проверки (пункт 4 задания)
learning_rates = [0.001, 0.01, 0.5, 1.0]
plt.figure( figsize=(15, 10) )

for i, lr in enumerate(learning_rates):
    model = Perceptron(input_dim=2)
    # Увеличим количество эпох до 2000 и добавим импульс beta = 0.9
    model.fit( X_train, y_train, X_test, y_test, epochs = 2000, lr = lr, batch_size = 32, beta = 0.9 )
    
    # Считаем точность
    metrics = get_metrics(y_test, model.predict(X_test))
    
    # Рисуем подграфик
    plt.subplot(2, 2, i+1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', alpha=0.6)
    
    # Отрисовка прямой
    x_range = np.array([X_test[:, 0].min(), X_test[:, 0].max()])
    # Важно: если w2 близко к 0, прямая может улететь, добавим проверку
    if abs(model.w[1]) > 1e-5:
        y_range = -(model.w[0] * x_range + model.b) / model.w[1]
        plt.plot(x_range, y_range, 'k--', lw=2)
    
    plt.title(f"LR = {lr}, Accuracy = {metrics['Accuracy']:.2f}")
    plt.ylim(X_test[:, 1].min()-0.5, X_test[:, 1].max()+0.5)

plt.tight_layout()
plt.show()
