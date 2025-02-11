import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from lerning.ML_.models.accuracy import accuracy
import random
import time

def data(n):
    # Генерация данных
    n = 25
    adults = [[round(random.uniform(1.4, 1.85), 2), round(random.uniform(50, 100), 1), round(random.uniform(5, 9), 2)] for _ in range(n)]
    children = [[round(random.uniform(0.9, 1.8), 2), round(random.uniform(30, 90), 1), round(random.uniform(7, 11), 2)] for _ in range(n)]

    # Подготовка данных
    X = np.array(adults + children)
    Y = np.array([0] * n + [1] * n)

    combined = list(zip(X, Y))
    random.shuffle(combined)
    X, Y = zip(*combined)
    X = np.array(X)
    Y = np.array(Y)

    # Разбиение данных
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test

# Работа с моделями

model_line = LinearRegression()
model_logistic = LogisticRegression()

accuracy_line, accuracy_logistic, time1_total, time2_total, kol_iteration = 0, 0, 0, 0, 0

for n in range(400, 1000):
    kol_iteration += 1
    X_train, X_test, Y_train, Y_test = data(n)

    time1 = time.time()
    model_line.fit(X_train, Y_train)
    predict_line = model_line.predict(X_test)
    accuracy_line += accuracy(predict_line, Y_test)
    time1_total += time.time() - time1



    time2 = time.time()
    model_logistic.fit(X_train, Y_train)
    predict_logistic = model_logistic.predict(X_test)
    accuracy_logistic += accuracy(predict_logistic, Y_test)
    time2_total += time.time() - time2


print("Line regration")
print(f"Accurancy: {accuracy_line / kol_iteration}; Time: {time1_total}", '\n')
print("Logistic regration")
print(f"Accurancy: {accuracy_logistic / kol_iteration}; Time: {time2_total}", '\n')

'''
Пробовал немного менять данные, увеличивать размер данных, всегда получается, что линейная регрессия работает гораздо 
быстрее, и точность показывается чуть выше. Попозже попробую узнать почему так и в каких случаях логистическая регрессия
показывает лучшие результаты.
'''






