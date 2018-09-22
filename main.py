from math import log, e, pow
import numpy as np


def main():
    data = np.genfromtxt('training_data.csv', delimiter=',', dtype='int32')
    x, y = data[:, 0], data[:, 1]

    eta = 0.1

    w = np.random.uniform(-1, 1, 1)

    for this_x in x:
        pred = logistic_fn(w, this_x)
        print(pred)
    print()
    t = 100000

    for _ in range(t):
        g = calculate_gradient(w, x, y)
        w = w - eta * g
        if g < 0.000001:
            break

    for this_x in x:
        pred = logistic_fn(w, this_x)
        print(pred)


def calculate_gradient(w, x, y):
    gradient_sum = 0
    for this_x, this_y in zip(x, y):
        gradient_sum += partial_gradient(w, this_x, this_y)
    return - (gradient_sum / len(x))


def partial_gradient(w, x, y):
    return (y * x) / (1 + pow(e, y * (np.dot(w, x))))


def logistic_fn(w, x):
    s = w * x
    return pow(e, s) / (1 + pow(e, s))


if __name__ == '__main__':
    main()
