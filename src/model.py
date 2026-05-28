import numpy as np

class Perceptron:
    def __init__( self, input_dim ):
        # Инициализация весов маленькими случайными значениями (пункт 2 задания)
        self.w = np.random.randn( input_dim ) * 0.01
        self.b = 0.0 # Смещение ноль
        self.train_loss_history = []
        self.val_loss_history = []

    def sigmoid( self, z ):
        # Сигмоида превращает любое число в вероятность от 0 до 1
        return 1 / ( 1 + np.exp( -np.clip( z, -500, 500 ) ) ) # clip защищает от переполнения exp

    def forward( self, X ):
        # z = w*x + b 
        z = np.dot( X, self.w ) + self.b
        return self.sigmoid( z )

    def compute_loss( self, y_true, y_pred ):
        # Бинарная кросс-энтропия (Log-loss)
        m = len( y_true )
        # Добавляем 1e-15, чтобы log(0) не выдал ошибку
        loss = -1 / m * np.sum( y_true * np.log( y_pred + 1e-15 ) + ( 1 - y_true ) * np.log( 1 - y_pred + 1e-15 ) )
        return loss

    def fit( self, X_train, y_train, X_val, y_val, epochs = 100, lr = 0.1, batch_size = 32, beta = 0.9, lambda_ = 0.01 ):
        m = X_train.shape[0]
        # Инициализируем скорости для момента (бонусное задание №4)
        v_w = np.zeros_like( self.w )
        v_b = 0

        for epoch in range( epochs ):
            indices = np.random.permutation( m )
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range( 0, m, batch_size ):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # 1. Прямой ход (предсказание)
                y_pred_batch = self.forward( X_batch )
                error = y_pred_batch - y_batch

                # Градиент с учётом L2-регуляции (бонусное задание №2)
                dw = ( 1 / len( X_batch ) ) * np.dot( X_batch.T, error ) + lambda_ * self.w
                db = ( 1 / len( X_batch ) ) * np.sum( error )

                # Применяем импульс (Momentum)
                v_w = beta * v_w + ( 1 - beta ) * dw
                v_b = beta * v_b + ( 1 - beta ) * db

                self.w -= lr * v_w
                self.b -= lr * v_b

            # Логируем лосс раз в 10 эпох
            if epoch % 10 == 0:
                self.train_loss_history.append( self.compute_loss( y_train, self.forward( X_train ) ) )
                self.val_loss_history.append( self.compute_loss( y_val, self.forward( X_val ) ) )





            # Каждые 100 эпох выводится лосс, чтобы видеть прогресс
            # if epoch % 100 == 0:
            #     train_loss = self.compute_loss( y_train, self.forward( X_train ) )
            #     self.train_loss_history.append( train_loss )
            #     self.val_loss_history.append( self.compute_loss( y_val, self.forward( X_val ) ) )


    # def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
    #     m = X_train.shape[0]
        
    #     for epoch in range(epochs):
    #         # Перемешивание (пункт 4 задания)
    #         indices = np.random.permutation(m)
    #         X_shuffled = X_train[indices]
    #         y_shuffled = y_train[indices]
            
    #         # Разделение на мини-батчи
    #         for i in range(0, m, batch_size):
    #             X_batch = X_shuffled[i:i+batch_size]
    #             y_batch = y_shuffled[i:i+batch_size]
                
    #             # 1. Прямой ход (предсказание)
    #             y_pred_batch = self.forward(X_batch)
                
    #             # 2. Считаем градиенты (формулы из раздела 3.1 теории)
    #             # Ошибка (y_hat - y)
    #             error = y_pred_batch - y_batch
    #             dw = (1 / len(X_batch)) * np.dot(X_batch.T, error)
    #             db = (1 / len(X_batch)) * np.sum(error)
                
    #             # 3. Обновление параметров (шаг градиентного спуска)
    #             self.w -= lr * dw
    #             self.b -= lr * db
            
    #         # Сохраняем потери для графиков
    #         train_loss = self.compute_loss(y_train, self.forward(X_train))
    #         val_loss = self.compute_loss(y_val, self.forward(X_val))
    #         self.train_loss_history.append(train_loss)
    #         self.val_loss_history.append(val_loss)

    def predict(self, X):
        # Если вероятность >= 0.5, то это класс 1, иначе 0
        return (self.forward(X) >= 0.5).astype(int)
    