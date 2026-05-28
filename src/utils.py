import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.model import Perceptron

def standard_scaler(X_train, X_test):
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (X_train - mean) / std, (X_test - mean) / std

def get_metrics(y_true, y_pred, y_prob=None):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    f1 = 2 * prec * rec / (prec + rec + 1e-7)
    
    res = {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1}
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        res["AUC"] = auc(fpr, tpr)
        res["roc"] = (fpr, tpr)
        
    return res

def cross_validate(X, y, epochs=100, lr=0.1, k=5):
    indices = np.random.permutation(len(y))
    folds = np.array_split(indices, k)
    scores = []
    
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        
        model = Perceptron(X.shape[1])
        model.fit(X_t, y_t, X_v, y_v, epochs=epochs, lr=lr)
        scores.append(np.mean(model.predict(X_v) == y_v))
        
    return np.mean(scores), np.std(scores)