import numpy as np

def standard_scaler(X_train, X_test):
    """
    Z-нормализация: (x - mean) / std.
    Важно: параметры (mean, std) считаются только по обучающей выборке!
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Чтобы избежать деления на ноль, если std = 0
    std = np.where(std == 0, 1, std)
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled

def get_metrics(y_true, y_pred):
    """
    Расчет метрик качества согласно разделу 6 теорет. дополнения.
    """
    # True Positives, False Positives, etc.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Добавляем маленькое число 1e-7, чтобы не делить на ноль
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }