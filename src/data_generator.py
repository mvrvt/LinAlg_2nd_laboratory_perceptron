import numpy as np

def generate_linear_data(n_samples=500, noise=0.1):
    # Два Гауссовых облака
    n_per_class = n_samples // 2
    cluster1 = np.random.randn(n_per_class, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_per_class, 2) + np.array([-2, -2])
    
    X = np.vstack([cluster1, cluster2])
    y = np.array([1]*n_per_class + [0]*n_per_class)
    
    # Добавляем шум (инверсия меток с вероятностью p)
    if noise > 0:
        mask = np.random.rand(n_samples) < noise
        y[mask] = 1 - y[mask]
        
    return X, y

def generate_circles_data(n_samples=500, factor=0.5, noise=0.05):
    # Внутренний и внешний круг (нелинейно разделимые)
    n_per_class = n_samples // 2
    iters = np.linspace(0, 2 * np.pi, n_per_class)
    
    # Внешний круг
    outer_x = np.cos(iters) + np.random.randn(n_per_class) * noise
    outer_y = np.sin(iters) + np.random.randn(n_per_class) * noise
    
    # Внутренний круг
    inner_x = np.cos(iters) * factor + np.random.randn(n_per_class) * noise
    inner_y = np.sin(iters) * factor + np.random.randn(n_per_class) * noise
    
    X = np.vstack([np.column_stack([outer_x, outer_y]), 
                   np.column_stack([inner_x, inner_y])])
    y = np.array([0]*n_per_class + [1]*n_per_class)
    return X, y
