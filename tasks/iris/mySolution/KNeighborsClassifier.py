from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np



class MyKNeighborsClassifier:
    def __init__(self, kol_N):
        self.kol_n = kol_N

    def fit(self, sings, target):
        self.sings = sings
        self.target = target

    def calculationDistance(self, sign, chekingObject):
        return np.sqrt(np.sum((np.array(sign) - np.array(chekingObject)) ** 2))

    def findNeighbors(self, chekingObject):
        mas = []
        for sign, target in zip(self.sings, self.target):
            distance = self.calculationDistance(sign, chekingObject)
            mas.append([distance, target])

        return mas

    def identification_of_results(self, chekingObject):
        mas = self.findNeighbors(chekingObject)
        mas_res = sorted(mas, key=lambda x: x[0])
        mas_Neig = mas_res[:self.kol_n]
        mas_classes = [note[1] for note in mas_Neig]

        return mas_classes

    def predict(self, signs_test):
        predict_mas = []
        for chekingObject in signs_test:
            mas_classes = self.identification_of_results(chekingObject)
            flag = dict()
            for i in mas_classes:
                if i not in flag:
                    flag[i] = 1
                else:
                    flag[i] += 1
            predict = -1
            maxValue = 0
            for key, value in flag.items():
                if maxValue < value:
                    predict = key
                    maxValue = value
            predict_mas.append(predict)

        return np.array(predict_mas)


def findMistake(X, Y):
    try:
        mistake = dict()
        for i in set(Y):
            mistake[f"Ирис типа {int(i)}"] = 0
        for x, y in zip(X, Y):
            if x != y:
                mistake[f"Ирис типа {int(y)}"] += 1
        print(f"Ошибки: {mistake}")
    except Exception as e:
        print(f"Ошибка в findMistake: {e}")


def fKNeighborsClassifier(X_train, X_test, y_train, y_test):

    model = MyKNeighborsClassifier(kol_N=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)