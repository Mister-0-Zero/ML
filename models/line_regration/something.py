import random

# MSE

def loss_function(mass_value, mass_predict, n):
    value_f = 0
    for y, y_ in zip(mass_value, mass_predict):
        value_f += (y - y_) ** 2
    value_f /= n
    return value_f

n = random.randint(5, 15)
mass_val = [random.randint(0, 1) for _ in range(n)]
mass_predict = [round(random.uniform(0.001, 0.999), 3) for _ in range(n)]
loss_val = loss_function(mass_val, mass_predict, n)

print(mass_val, mass_predict, loss_val, sep='\n')