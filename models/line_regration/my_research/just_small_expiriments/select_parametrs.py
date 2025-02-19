from lerning.ML_.models.line_regration.my_research.just_small_expiriments.my_model import LinearRegressionGD
from lerning.ML_.models.accuracy import accuracy

def select_param(X_train, X_test, y_train, y_test):
    best_param = [0, 0]
    best_accuracy = 0
    try:
        for i in range(1, 200, 40):
            for j in range(20, 1000, 70):
                model = LinearRegressionGD(i / 10000, j)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                accuracy_ = accuracy(predict, y_test)
                if accuracy_ > best_accuracy:
                    best_accuracy = accuracy_
                    best_param = [i / 10000, j]
    except:
        pass

    return best_param


