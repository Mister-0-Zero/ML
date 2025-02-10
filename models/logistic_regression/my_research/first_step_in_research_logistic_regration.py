import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random

# Генерация данных
n = 25
adults = [[round(random.uniform(1.5, 1.85), 2), round(random.uniform(65, 100), 1), round(random.uniform(5, 9), 2)] for _ in range(n)]
children = [[round(random.uniform(0.9, 1.8), 2), round(random.uniform(30, 90), 1), round(random.uniform(7, 11), 2)] for _ in range(n)]

X = np.array(adults + children)
Y = np.array([0] * n + [1] * n)  # 0 - взрослые, 1 - дети

# Перемешивание данных
combined = list(zip(X, Y))
random.shuffle(combined)
X, Y = zip(*combined)
X = np.array(X)
Y = np.array(Y)

# Разделение на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Обучение логистической регрессии
model = LogisticRegression()
model.fit(X_train, Y_train)
predict = model.predict(X_test)

# Преобразуем Y обратно в массив и делим данные по классам
Y = np.array(Y)
X_adults = X[Y == 0]
X_children = X[Y == 1]

# Создание графиков
fig = plt.figure(figsize=(12, 8))

# 3D-график
ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title("3D: Рост, Вес, Длительность сна")
ax1.set_xlabel("Рост (м)")
ax1.set_ylabel("Вес (кг)")
ax1.set_zlabel("Сон (ч)")

ax1.scatter(X_adults[:, 0], X_adults[:, 1], X_adults[:, 2], color='blue', label='Взрослые (обучение)')
ax1.scatter(X_children[:, 0], X_children[:, 1], X_children[:, 2], color='red', label='Дети (обучение)')

# Тестовые предсказания
for i, point in enumerate(X_test):
    color = 'green' if predict[i] == 0 else 'orange'
    ax1.scatter(point[0], point[1], point[2], color=color, marker='x', s=100)

ax1.legend()

# Функция для построения 2D-графиков
def TwoD(subplot, title, xlabel, ylabel, idx_x, idx_y):
    ax = fig.add_subplot(subplot)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.scatter(X_adults[:, idx_x], X_adults[:, idx_y], color='blue', label='Взрослые')
    ax.scatter(X_children[:, idx_x], X_children[:, idx_y], color='red', label='Дети')

    for i, point in enumerate(X_test):
        color = 'green' if predict[i] == 0 else 'orange'
        ax.scatter(point[idx_x], point[idx_y], color=color, marker='x', s=100)

    ax.legend()

# Построение 2D-графиков
TwoD(222, "Рост и вес", "Рост (м)", "Вес (кг)", 0, 1)
TwoD(223, "Рост и длительность сна", "Рост (м)", "Длительность сна (ч)", 0, 2)
TwoD(224, "Вес и длительность сна", "Вес (кг)", "Длительность сна (ч)", 1, 2)

plt.tight_layout()
plt.show()
