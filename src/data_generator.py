import numpy as np

def generate_linear_data(n_samples=500, noise=0.05):
    n_per_class = n_samples // 2
    cluster1 = np.random.randn(n_per_class, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_per_class, 2) + np.array([-2, -2])
    X = np.vstack([cluster1, cluster2])
    y = np.array([1]*n_per_class + [0]*n_per_class)
    if noise > 0:
        mask = np.random.rand(n_samples) < noise
        y[mask] = 1 - y[mask]
    return X, y

def generate_xor_data(n_samples=500):
    # Точки в четырех квадрантах (XOR)
    X = np.random.uniform(-2, 2, (n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y

def generate_circles_data(n_samples=500, factor=0.5):
    n_per_class = n_samples // 2
    angles = np.linspace(0, 2*np.pi, n_per_class)
    outer = np.column_stack([np.cos(angles), np.sin(angles)]) + np.random.randn(n_per_class, 2) * 0.1
    inner = outer * factor + np.random.randn(n_per_class, 2) * 0.1
    X = np.vstack([outer, inner])
    y = np.array([0]*n_per_class + [1]*n_per_class)
    return X, y
