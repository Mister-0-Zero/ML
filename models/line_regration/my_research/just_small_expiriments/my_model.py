import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate #задает скорость обучения
        self.epochs = epochs #задает количество итераций для обучения
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        try:
            n_samples, n_features = X.shape # n_sample - количество строк (объектов) и n_features - количество признаков
            self.weights = np.zeros(n_features) # вектор весов
            self.bias = 0 # свободный коэффициент (смещение)

            for _ in range(self.epochs):
                y_predicted = np.dot(X, self.weights) + self.bias # разделяющая линия

                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # считаем на сколько надо изменить веса
                db = (1 / n_samples) * np.sum(y_predicted - y) # считаем на сколько надо изменить свободный член

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        except:
            pass

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
