import numpy as np

class Perceptron:
    def __init__(self, input_dim, init_type='small_random'):
        if init_type == 'zero':
            self.w = np.zeros(input_dim)
        elif init_type == 'large_random':
            self.w = np.random.randn(input_dim) * 10
        else: # small_random
            self.w = np.random.randn(input_dim) * 0.01
        self.b = 0
        self.train_loss_history = []
        self.val_loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def compute_loss(self, y_true, y_pred, loss_type='ce', lambda_=0):
        m = len(y_true)
        if loss_type == 'ce':
            loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        elif loss_type == 'hinge':
            # Hinge требует y из {-1, 1}
            y_h = np.where(y_true == 0, -1, 1)
            loss = np.mean(np.maximum(0, 1 - y_h * (np.dot(y_pred, 2) - 1))) # Упрощенно
            
        # L2 Regularization
        reg_term = (lambda_ / (2 * m)) * np.sum(self.w**2)
        return loss + reg_term

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, 
            batch_size=32, beta=0.9, lambda_=0.01, loss_type='ce'):
        m = X_train.shape[0]
        v_w, v_b = np.zeros_like(self.w), 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_s, y_s = X_train[indices], y_train[indices]
            
            for i in range(0, m, batch_size):
                Xi, yi = X_s[i:i+batch_size], y_s[i:i+batch_size]
                y_p = self.forward(Xi)
                
                # Градиенты
                error = y_p - yi
                dw = (1/len(Xi)) * np.dot(Xi.T, error) + (lambda_/m) * self.w
                db = (1/len(Xi)) * np.sum(error)
                
                # Optimizer: Momentum vs SGD
                if beta > 0:
                    v_w = beta * v_w + (1-beta) * dw
                    v_b = beta * v_b + (1-beta) * db
                    self.w -= lr * v_w
                    self.b -= lr * v_b
                else:
                    self.w -= lr * dw
                    self.b -= lr * db
            
            self.train_loss_history.append(self.compute_loss(y_train, self.forward(X_train), loss_type, lambda_))
            self.val_loss_history.append(self.compute_loss(y_val, self.forward(X_val), loss_type, lambda_))

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)
    
