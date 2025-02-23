import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Данные (Возраст, Опыт (в годах), Знает Python? (1 - да, 0 - нет))
X = np.array([
    [25, 3, 1],
    [30, 5, 1],
    [22, 1, 0],
    [40, 10, 1],
    [35, 7, 0],
    [29, 4, 0]
])

# Целевая переменная (0 - не подходит, 1 - подходит)
y = np.array([1, 1, 0, 1, 0, 0])

# Создаём и обучаем модель
model = DecisionTreeClassifier()
model.fit(X, y)

# Визуализация дерева решений
plt.figure(figsize=(10, 6))  # Размер графика
plot_tree(model, feature_names=["Возраст", "Опыт", "Знает Python?"],
          class_names=["Не подходит", "Подходит"], filled=True)

plt.show()  # Отобразить график
# Прогнозируем, подходит ли кандидат (27 лет, 3 года опыта, знает Python)
sample_candidate = np.array([[27, 3, 1]])
prediction = model.predict(sample_candidate)

print("Результат:", "Подходит" if prediction[0] == 1 else "Не подходит")

