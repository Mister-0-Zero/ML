import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Генерируем случайные данные
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Признаки
y = 4 + 3 * X + np.random.randn(100, 1)  # Целевая переменная (с шумом)

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Визуализация
plt.scatter(X_test, y_test, color='blue', label="Реальные данные")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Линейная регрессия")
plt.legend()
plt.show()

# Вывод коэффициентов
print("Коэффициенты модели:", model.coef_, model.intercept_)


'''
Что тут происходит

мы генерируем случайные данные, признаки и значения (строки 8, 9), создаем исскуственно зависимость 4 + 3 * X
мы сгенирировали признаки и заполнили Y по закону y = 4 + 3 * x

график отражает линию найденную алгоритмом, ниже приведен найденный коэффициент наклона и свободный коэффициент
'''