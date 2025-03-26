from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time

from mySolution.KNeighborsClassifier import fKNeighborsClassifier
from internet_and_AI_solution.gptSolution import gptSolution
from internet_and_AI_solution.random_foerst import random_forest

from additional_functionality_and_reports_results.print_long_string import print_long_string
from additional_functionality_and_reports_results.reports_results import *


mas_name = ["fKNeighborsClassifier", "gptSolution", "random_forest"]
func = [fKNeighborsClassifier, gptSolution, random_forest]

data = load_iris()
X = data.data
y = data.target

print_long_string(description)
passge_number = 0

if __name__ == "__main__":
    for name, function in zip(mas_name, func):
        passge_number += 1
        n, accurancy, time_start = 10, 0, time.time()

        print(name)

        for i in range(n):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
            accurancy += function(X_train, X_test, y_train, y_test)
        time_end = time.time()

        print("Accurancy: ", accurancy / n)
        print(f"Time = {time_end - time_start}")

        if passge_number % 2 == 0:
            if passge_number == 2: print_long_string(res1)
        else:
            print()



