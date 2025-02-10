import random
import numpy as np


def loss_function(mas_value, mas_prediction, n):
    value_f = 0
    for y, y_ in zip(mas_value, mas_prediction):
        value_f += y * np.log(y_) + (1 - y) * np.log(1 - y_)
    value_f /= -n
    return value_f

n = random.randint(5, 15)
mass_y = [random.randint(0, 1) for _ in range(n)]
mass_y_ = [round(random.uniform(0.01, 0.99), 3) for _ in range(n)]

value_loss_function = loss_function(mass_y, mass_y_, n)
value_loss_function2 = loss_function([1], [0.5], 1)

print(mass_y, mass_y_, value_loss_function, sep="\n")
print()
print(value_loss_function2)
