import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.model import Perceptron
from src.utils import standard_scaler, get_metrics

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_test = standard_scaler(X_train, X_test)

# --- Эксперимент 1: Скорость обучения (Learning Rate) ---
learning_rates = [0.001, 0.01, 0.5, 1.0]
print("\n=== Таблица 1: Влияние Learning Rate ===")
plt.figure(figsize=(12, 10))
for i, lr in enumerate(learning_rates):
    model = Perceptron(input_dim=2, init_type='small_random')
    model.fit(X_train, y_train, X_test, y_test, epochs=150, lr=lr, batch_size=32)
    t_acc = get_metrics(y_train, model.predict(X_train))['Accuracy']
    v_acc = get_metrics(y_test, model.predict(X_test))['Accuracy']
    print(f"LR: {lr:<5} | Точность на трейне: {t_acc:.4f} | Точность на тесте: {v_acc:.4f}")
    
    plt.subplot(2, 2, i+1)
    plt.plot(model.train_loss_history, label='Train')
    plt.plot(model.val_loss_history, label='Val')
    plt.title(f"LR = {lr}")
    plt.legend()
plt.suptitle("Влияние Learning Rate на сходимость функции потерь", fontsize=14)
plt.tight_layout()
plt.show()

# --- Эксперимент 2: Размер батча (Batch Size) ---
batch_sizes = [1, 16, 64, 256]
print("\n=== Таблица 2: Влияние размера батча ===")
plt.figure(figsize=(12, 10))
for i, bs in enumerate(batch_sizes):
    model = Perceptron(input_dim=2, init_type='small_random')
    model.fit(X_train, y_train, X_test, y_test, epochs=150, lr=0.1, batch_size=bs)
    t_acc = get_metrics(y_train, model.predict(X_train))['Accuracy']
    v_acc = get_metrics(y_test, model.predict(X_test))['Accuracy']
    print(f"Batch Size: {bs:<3} | Точность на трейне: {t_acc:.4f} | Точность на тесте: {v_acc:.4f}")
    
    plt.subplot(2, 2, i+1)
    plt.plot(model.train_loss_history, label='Train')
    plt.plot(model.val_loss_history, label='Val')
    plt.title(f"Batch Size = {bs}")
    plt.legend()
plt.suptitle("Влияние размера батча на сходимость функции потерь", fontsize=14)
plt.tight_layout()
plt.show()

# --- Эксперимент 3: Инициализация весов ---
init_types = ['zero', 'small_random', 'large_random']
print("\n=== Таблица 3: Влияние инициализации весов ===")
plt.figure(figsize=(15, 5))
for i, init_t in enumerate(init_types):
    model = Perceptron(input_dim=2, init_type=init_t)
    model.fit(X_train, y_train, X_test, y_test, epochs=150, lr=0.1, batch_size=32)
    t_acc = get_metrics(y_train, model.predict(X_train))['Accuracy']
    v_acc = get_metrics(y_test, model.predict(X_test))['Accuracy']
    print(f"Инициализация: {init_t:<13} | Трейн: {t_acc:.4f} | Тест: {v_acc:.4f}")
    
    plt.subplot(1, 3, i+1)
    plt.plot(model.train_loss_history, label='Train')
    plt.plot(model.val_loss_history, label='Val')
    plt.title(f"Тип: {init_t}")
    plt.legend()
plt.suptitle("Влияние инициализации весов на сходимость функции потерь", fontsize=14)
plt.tight_layout()
plt.show()
