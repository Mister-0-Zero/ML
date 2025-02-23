from DecisionTreeClassifier import MyDecisionTree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

model = MyDecisionTree()
model.fit(X_train, Y_train)
predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
print(accuracy)