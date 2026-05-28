import numpy as np

def generate_linear_data(n_samples=500, noise=0.05, cov=[[1.0, 0.0], [0.0, 1.0]]):
    """Генерация двух гауссовых облаков с заданной матрицей ковариации"""
    n_per_class = n_samples // 2
    cov_matrix = np.array(cov)
    
    cluster1 = np.random.multivariate_normal([2.0, 2.0], cov_matrix, n_per_class)
    cluster2 = np.random.multivariate_normal([-2.0, -2.0], cov_matrix, n_per_class)
    
    X = np.vstack([cluster1, cluster2])
    y = np.array([1]*n_per_class + [0]*n_per_class)
    
    if noise > 0:
        mask = np.random.rand(n_samples) < noise
        y[mask] = 1 - y[mask]
    return X, y

def generate_xor_data(n_samples=500, noise=0.0):
    """Точки в четырех квадрантах (XOR задача)"""
    X = np.random.uniform(-2, 2, (n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    if noise > 0:
        mask = np.random.rand(n_samples) < noise
        y[mask] = 1 - y[mask]
    return X, y

def generate_circles_data(n_samples=500, factor=0.5, noise=0.0):
    """Точки внутри и снаружи окружности"""
    n_per_class = n_samples // 2
    angles = np.linspace(0, 2 * np.pi, n_per_class)
    outer = np.column_stack([np.cos(angles), np.sin(angles)]) + np.random.randn(n_per_class, 2) * 0.1
    inner = outer * factor + np.random.randn(n_per_class, 2) * 0.1
    
    X = np.vstack([outer, inner])
    y = np.array([0]*n_per_class + [1]*n_per_class)
    if noise > 0:
        mask = np.random.rand(n_samples) < noise
        y[mask] = 1 - y[mask]
    return X, y
