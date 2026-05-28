import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def standard_scaler(X_train, X_test):
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / (std + 1e-7), (X_test - mean) / (std + 1e-7)

def get_metrics(y_true, y_pred, y_prob=None):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics = {
        "Accuracy": (tp + tn) / len(y_true),
        "Precision": tp / (tp + fp + 1e-7),
        "Recall": tp / (tp + fn + 1e-7),
        "F1": 2 * tp / (2 * tp + fp + fn + 1e-7)
    }
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["auc"] = auc(fpr, tpr)
        metrics["roc"] = (fpr, tpr)
    return metrics

def cross_validate(X, y, epochs=50, lr=0.1, batch_size=32, beta=0.9, lambda_=0.01, loss_type='ce', k=5):
    from src.model import Perceptron
    indices = np.random.permutation(len(y))
    folds = np.array_split(indices, k)
    scores = []
    
    for i in range(k):
        v_idx = folds[i]
        t_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        model = Perceptron(X.shape[1])
        model.fit(X[t_idx], y[t_idx], X[v_idx], y[v_idx], 
                  epochs=epochs, lr=lr, batch_size=batch_size, beta=beta, lambda_=lambda_, loss_type=loss_type)
        scores.append(np.mean(model.predict(X[v_idx]) == y[v_idx]))
        
    return np.mean(scores), np.std(scores)
