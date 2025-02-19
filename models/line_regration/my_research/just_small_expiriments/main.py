import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
from my_model import LinearRegressionGD
from lerning.ML_.models.accuracy import accuracy
from select_parametrs import select_param

# Генерация данных
n = 30
X0 = [[round(random.uniform(6, 15), 1), round(random.uniform(1.2, 3), 1)] for _ in range(n)]
X1 = [[round(random.uniform(9, 21), 1), round(random.uniform(2.0, 5), 1)] for _ in range(n)]

X, Y = [], []
x0_score, x1_score = 0, 0

while x0_score < len(X0) and x1_score < len(X1):
    if random.randint(0, 1):
        X.append(X0[x0_score])
        Y.append(0)
        x0_score += 1
    else:
        X.append(X1[x1_score])
        Y.append(1)
        x1_score += 1

def add_remaining_samples(index, X_, label):
    while index < len(X_):
        X.append(X_[index])
        Y.append(label)
        index += 1

add_remaining_samples(x0_score, X0, 0)
add_remaining_samples(x1_score, X1, 1)

# Преобразуем X в numpy массив
X = np.array(X)
Y = np.array(Y)

print(f"n: {n}")
print(f"The first 5 signs X0: {X0[:5]}")
print(f"The first 5 signs X1: {X1[:5]}")
print(f"X: {X[:5]}")
print(f"Y: {Y[:5]}")

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# Обучение моделей
def train_models():
    global modelGPT, my_model, y_predGPT, my_y_pred
    modelGPT = LinearRegression()
    modelGPT.fit(X_train, y_train)
    y_predGPT = modelGPT.predict(X_test)

    step, kol_iteration = select_param(X_train, X_test, y_train, y_test)
    my_model = LinearRegressionGD(step, kol_iteration)
    my_model.fit(X_train, y_train)
    my_y_pred = my_model.predict(X_test)


train_models()


def application_scatter(ax, X, color, label="", s=60):
    if len(X) > 0:
        X = np.array(X)
        ax.scatter(X[:, 0], X[:, 1], color=color, s=s, label=label)


# Функция для обновления графика
def update_plot():
    axs[0].cla()
    axs[1].cla()

    # Первый график
    axs[0].set_title("GPT solution")
    axs[0].set_xlabel("Height")
    axs[0].set_ylabel("Radius_branch")
    application_scatter(axs[0], X[Y == 0], "blue", "X0 trees")
    application_scatter(axs[0], X[Y == 1], "red", "X1 trees")
    application_scatter(axs[0], X_test[y_predGPT >= 0.5], (0.5, 1, 0.5), "Predicted X1 trees", 35)
    application_scatter(axs[0], X_test[y_predGPT < 0.5], (0.5, 0.0, 0.5), "Predicted X0 trees", 35)
    axs[0].legend()
    axs[0].text(0.5, -0.13, f"accuracy: {accuracy(y_predGPT, y_test)}", ha='center', transform=axs[0].transAxes)

    # Второй график
    axs[1].set_title("My solution")
    axs[1].set_xlabel("Height")
    axs[1].set_ylabel("Radius_branch")
    application_scatter(axs[1], X[Y == 0], "blue", "X0 trees")
    application_scatter(axs[1], X[Y == 1], "red", "X1 trees")
    application_scatter(axs[1], X_test[my_y_pred >= 0.5], (0.5, 1, 0.5), "Predicted X1 trees", 35)
    application_scatter(axs[1], X_test[my_y_pred < 0.5], (0.5, 0.0, 0.5), "Predicted X0 trees", 35)
    axs[1].legend()
    axs[1].text(0.5, -0.13, f"accuracy: {accuracy(my_y_pred, y_test)}", ha='center', transform=axs[1].transAxes)
    plt.draw()


# Функция для обработки кликов
def on_click(event):
    global X_train, y_train, X_test, y_test, X, Y
    if event.inaxes is None:
        return

    new_point = [event.xdata, event.ydata]
    label = 0 if event.button == 1 else 1  # Левая кнопка - синий, правая - красный

    X_train = np.vstack([X_train, new_point])
    y_train = np.append(y_train, label)

    X = np.vstack([X, new_point])
    Y = np.append(Y, label)

    # Переобучаем модель
    train_models()
    update_plot()


# Создаем область для двух графиков
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.canvas.mpl_connect("button_press_event", on_click)
update_plot()
plt.show()

'''
# y_line = w0 + w1 * x_values + w2 * fixed_radius
w0 = model.intercept_
w1, w2 = model.coef_

height = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
radius_branch = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

y = model.predict([[height[i], radius_branch[i]] for i in range(100)])

plt.plot(?, ?, color='green', linewidth=2, label="Regression Line")
'''
