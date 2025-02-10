import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import math

def gptSolution(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)

    # cm = confusion_matrix(y_test, y_pred)
    #
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    # disp.plot(cmap='Blues')
    #
    # plt.show()
