import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv')
data = np.array(data)

xs = data[:, 0]
ys = data[:, 1]


def h(x, t0, t1):
    return t0 + t1 * x


def cost_function(xs, ys, t0, t1, m):
    return (1 / (2 * m)) * sum_err(xs, ys, t0, t1, m)


def sum_err(xs, ys, t0, t1, m):
    total = 0
    for i in range(m):
        total += (h(xs[i], t0, t1) - ys[i])**2

    return total


def derivative_t0(xs, ys, t0, t1, m):
    return (1 / m) * sum_err_t0(xs, ys, t0, t1, m)


def derivative_t1(xs, ys, t0, t1, m):
    return (1 / m) * sum_err_t1(xs, ys, t0, t1, m)


def sum_err_t0(xs, ys, t0, t1, m):
    total = 0
    for i in range(m):
        total += (h(xs[i], t0, t1) - ys[i])

    return total


def sum_err_t1(xs, ys, t0, t1, m):
    total = 0
    for i in range(m):
        total += (h(xs[i], t0, t1) - ys[i]) * xs[i]

    return total


def gradient_descent(xs, ys, m):
    t0 = 0
    t1 = 0
    learning_rate = 0.0001
    for i in range(0, 1000):
        temp0 = t0 - learning_rate * derivative_t0(xs, ys, t0, t1, m)
        temp1 = t1 - learning_rate * derivative_t1(xs, ys, t0, t1, m)
        t0 = temp0
        t1 = temp1
        err = cost_function(xs, ys, t0, t1, m)


#         print(err)

    return t0, t1

t0, t1 = gradient_descent(xs[:], ys[:], len(data))
print('--------------------------')
print(f't0 = {t0}, t1 = {t1}')

r = h(42, t0, t1)

# plt.xlim(-20, 120)
# plt.ylim(0, 140)
plt.scatter(xs, ys)
plt.plot(xs, [h(i, t0, t1) for i in xs], c='r')
plt.scatter(42, r, c='k', s=100)

plt.show()
