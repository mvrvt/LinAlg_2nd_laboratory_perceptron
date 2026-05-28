import numpy as np
import matplotlib.pyplot as plt
from src.model import Perceptron
from src.utils import standard_scaler

# 1. Генерация двух четких облаков точке (задание №1 на доп. баллы)
cluster1 = np.random.randn( 100, 2 ) + np.array( [2, 2] )   # центр (2, 2)
cluster2 = np.random.randn( 100, 2 ) + np.array( [-2, -2] ) # центр (-2, -2)

X = np.vstack( [cluster1, cluster2] ) 
y = np.array( [1] * 100 + [0] * 100 )

# Стандартизация 
X_scaled, _ = standard_scaler( X, X )

# 2. Обучение
model = Perceptron( input_dim = 2 )
# Даём LR побольше и включаем регуляризацию
model.fit( X_scaled, y, X_scaled, y, epochs = 500, lr = 0.5, lambda_ = 0.001 )

# 3. Визуализация 
plt.figure( figsize = ( 8, 6 ) )
plt.scatter( X_scaled[:, 0], X_scaled[:, 1], c = y, cmap = 'coolwarm', edgecolors = 'k' )

x_vals = np.linspace( X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100 )
y_vals = -( model.w[0] * x_vals + model.b ) / model.w[1]
plt.plot( x_vals, y_vals, 'black', lw=3, label='Идеальная граница' )
plt.title( f"Бонус: Линейно разделимые данные. Accuracy: {np.mean( model.predict( X_scaled ) == y )}" )
plt.legend()
plt.show()
