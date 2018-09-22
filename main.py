from math import e, pow, log
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def main():
    x_range = np.array(range(-10, 10))
    lr = LogisticRegression()

    data = np.genfromtxt('training_data.csv', delimiter=',', dtype='int32')
    x, y = data[:, 0], data[:, 1]
    x = np.column_stack((np.ones_like(x), x))  # Let x0 equal 1

    lr.fit(x, y)

    print(x)
    print(y)

    eta = 0.1

    w = np.random.uniform(-10, 10, 2)

    t = 200000

    for i in range(t):
        g = calculate_gradient(w, x, y)
        w = w - eta * g

        if np.sum(np.absolute(g)) < 0.0001 and i > 10000:
            print()
            print(i)
            break

    print()

    for this_x, this_y in zip(x, y):
        pred = logistic_fn(w, this_x, this_y)
        print(pred)


def calculate_gradient(w, x, y):
    gradient_sum = 0
    for this_x, this_y in zip(x, y):
        gradient_sum += partial_gradient(w, this_x, this_y)
    return - (gradient_sum / x.shape[0])


def partial_gradient(w, x, y):
    return (y * x) / (1 + pow(e, y * (np.dot(w, x))))


def logistic_fn(w, x, y):
    s = y * np.dot(w, x)
    return pow(e, s) / (1 + pow(e, s))


def insample_error(w, x, y):
    sum = 0
    for this_x, this_y in zip(x, y):
        sum += pt_error(w, this_x, this_y)
    return sum / x.shape[0]


def pt_error(w, x, y):
    return log(1 + pow(e, -y * np.dot(w, x)))


if __name__ == '__main__':
    main()
