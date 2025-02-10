# Импортируем нужные библиотеки:
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, linear_model
from sklearn.cluster import KMeans
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt  # Исправленный импорт

# Загружаем набор данных Ирисы:
iris = datasets.load_iris()

# Смотрим на названия переменных
print(iris.feature_names)

# Смотрим на данные, выводим 10 первых строк:
print(iris.data[:10])

# Смотрим на целевую переменную:
print(iris.target_names)
print(iris.target)

# Создаём DataFrame
iris_frame = DataFrame(iris.data)
iris_frame.columns = iris.feature_names  # Имена колонок
iris_frame['target'] = iris.target  # Целевая переменная
iris_frame['name'] = iris_frame.target.apply(lambda x: iris.target_names[x])  # Добавляем "name"

# Смотрим на DataFrame
print(iris_frame)

# Визуализация данных (если vvod != 0)
vvod = 0
if vvod:
    # Гистограммы
    plt.figure(figsize=(20, 24))
    plot_number = 0
    for feature_name in iris['feature_names']:
        for target_name in iris['target_names']:
            plot_number += 1
            plt.subplot(4, 3, plot_number)
            plt.hist(iris_frame[iris_frame.name == target_name][feature_name])
            plt.title(target_name)
            plt.xlabel('cm')
            plt.ylabel(feature_name[:-4])

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Pairplot
    sns.pairplot(iris_frame[['sepal length (cm)', 'sepal width (cm)',
                             'petal length (cm)', 'petal width (cm)', 'name']],
                 hue='name')
    plt.show()

# Разделение данных на обучающие и тестовые
train_data, test_data, train_labels, test_labels = train_test_split(
    iris_frame[['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)']],
    iris_frame[['target']],
    test_size=0.3,
    random_state=0
)

# Создание и обучение модели
model = linear_model.SGDClassifier(alpha=0.001, max_iter=100, random_state=0)
model.fit(train_data, train_labels.values.ravel())  # ravel для правильной формы

# Предсказание на тестовых данных
model_predictions = model.predict(test_data)

# Метрики качества модели
print("Accuracy:", metrics.accuracy_score(test_labels, model_predictions))
print(metrics.classification_report(test_labels, model_predictions))

# Кросс-валидация модели
scores = cross_val_score(model, train_data, train_labels.values.ravel(), cv=10)
print("Cross-validation accuracy:", scores.mean())
