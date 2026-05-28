import numpy as np
import matplotlib.pyplot as plt
from src.data_generator import generate_linear_data, generate_xor_data, generate_circles_data
from src.model import Perceptron
from src.utils import standard_scaler, get_metrics, cross_validate

print("================ БОНУСНЫЙ БЛОК ЗАДАНИЙ ================")

# --- Задание 1: Собственный генератор данных и границы применимости ---
plt.figure(figsize=(15, 4))
datasets = {
    "Линейно разделимые": generate_linear_data(noise=0.02, cov=[[0.8, 0.1], [0.1, 0.8]]),
    "Нелинейные: XOR": generate_xor_data(),
    "Нелинейные: Окружность": generate_circles_data()
}

for i, (name, (X_d, y_d)) in enumerate(datasets.items()):
    X_sc, _ = standard_scaler(X_d, X_d)
    m_d = Perceptron(2)
    m_d.fit(X_sc, y_d, X_sc, y_d, epochs=100, lr=0.1)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X_sc[:, 0], X_sc[:, 1], c=m_d.predict(X_sc), cmap='coolwarm', edgecolors='k', alpha=0.7)
    plt.title(f"{name}\nAccuracy: {np.mean(m_d.predict(X_sc) == y_d):.2f}")
plt.tight_layout()
plt.show()


# --- Задание 2: Дополнительные функции потерь и L2-регуляризация ---
X_l, y_l = generate_linear_data(n_samples=400)
X_l_train, X_l_val = standard_scaler(X_l[:280], X_l[280:])
y_l_train, y_l_val = y_l[:280], y_l[280:]

# Сравнение лоссов
m_ce = Perceptron(2)
m_ce.fit(X_l_train, y_l_train, X_l_val, y_l_val, epochs=100, loss_type='ce', lambda_=0)
m_hinge = Perceptron(2)
m_hinge.fit(X_l_train, y_l_train, X_l_val, y_l_val, epochs=100, loss_type='hinge', lambda_=0)

plt.figure(figsize=(10, 4))
plt.plot(m_ce.train_loss_history, label='Бинарная кросс-энтропия')
plt.plot(m_hinge.train_loss_history, label='Hinge Loss')
plt.title("Сравнение скорости сходимости функций потерь")
plt.legend()
plt.show()

# ИСПРАВЛЕНО И ДОРАБОТАНО: Исследование L2 регуляризации с выводом графиков
lambdas = [0.0, 0.01, 0.1, 1.0]
norms = []
val_accuracies = []

print("\nВлияние коэффициента регуляризации L2:")
for lam in lambdas:
    m_reg = Perceptron(2)
    m_reg.fit(X_l_train, y_l_train, X_l_val, y_l_val, epochs=100, lambda_=lam)
    w_norm = np.linalg.norm(m_reg.w)
    norms.append(w_norm)
    acc = np.mean(m_reg.predict(X_l_val) == y_l_val)
    val_accuracies.append(acc)
    print(f"Лямбда: {lam:<4} | Норма весов ||w||_2: {w_norm:.4f} | Точность на валидации: {acc:.4f}")

# Отрисовка графиков для L2-регуляризации
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lambdas, norms, marker='o', color='blue', linestyle='-', linewidth=2)
plt.xlabel('Коэффициент регуляризации (lambda_)')
plt.ylabel('Норма весов ||w||_2')
plt.title('Штраф весов при росте Lambda')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lambdas, val_accuracies, marker='s', color='green', linestyle='--', linewidth=2)
plt.xlabel('Коэффициент регуляризации (lambda_)')
plt.ylabel('Validation Accuracy')
plt.title('Влияние L2 на обобщающую способность')
plt.grid(True)

plt.tight_layout()
plt.show()


# --- Задание 3: Метрики качества и визуальный анализ ошибок ---
m_final = Perceptron(2)
m_final.fit(X_l_train, y_l_train, X_l_val, y_l_val, epochs=150, lr=0.1)
probs = m_final.forward(X_l_val)
preds = m_final.predict(X_l_val)
metrics = get_metrics(y_l_val, preds, probs)

print("\nПродвинутые метрики на валидации:")
for k, v in metrics.items():
    if k not in ['roc', 'auc']:
        print(f" - {k}: {v:.4f}")
print(f" - ROC-AUC: {metrics['auc']:.4f}")

# График ROC
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(metrics['roc'][0], metrics['roc'][1], label=f"AUC = {metrics['auc']:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC-кривая")
plt.legend()

# Визуализация ошибок классификации
plt.subplot(1, 2, 2)
incorrect = (preds != y_l_val)
plt.scatter(X_l_val[:, 0], X_l_val[:, 1], c=y_l_val, cmap='coolwarm', edgecolors='k', alpha=0.6)
if np.sum(incorrect) > 0:
    plt.scatter(X_l_val[incorrect, 0], X_l_val[incorrect, 1], facecolors='none', edgecolors='black', s=150, lw=2, label='Ошибки модели')
plt.title("Анализ ошибочных предсказаний")
plt.legend()
plt.tight_layout()
plt.show()


# --- Задание 4: Градиентный спуск с моментом (Momentum) ---
betas = [0.0, 0.5, 0.9, 0.99]
plt.figure()
for b in betas:
    m_b = Perceptron(2)
    m_b.fit(X_l_train, y_l_train, X_l_val, y_l_val, epochs=80, lr=0.05, beta=b)
    label_name = 'Обычный SGD' if b == 0.0 else f'Momentum (beta={b})'
    plt.plot(m_b.train_loss_history, label=label_name)
plt.title("Исследование сходимости алгоритма Momentum")
plt.legend()
plt.show()


# --- Задание 5: Кросс-валидация и автоматический подбор гиперпараметров ---
lrs = [0.01, 0.1, 0.5]
batch_sizes = [16, 32, 64]

best_score = -1
best_params = {}

print("\nЗапуск Grid Search 5-Fold Кросс-валидации:")
for lr in lrs:
    for bs in batch_sizes:
        mean_acc, std_acc = cross_validate(X_l_train, y_l_train, epochs=40, lr=lr, batch_size=bs, k=5)
        print(f"Параметры -> LR: {lr}, Batch: {bs} | Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        if mean_acc > best_score:
            best_score = mean_acc
            best_params = {'lr': lr, 'batch_size': bs}

print(f"\n[Успех] Лучшие параметры: {best_params} со средней точностью {best_score:.4f}")

# Обучение финальной модели на всей выборке
top_model = Perceptron(2)
top_model.fit(X_l_train, y_l_train, X_l_val, y_l_val, epochs=100, lr=best_params['lr'], batch_size=best_params['batch_size'])
print(f"Итоговая точность финальной модели: {np.mean(top_model.predict(X_l_val) == y_l_val):.4f}")
