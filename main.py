from math import e, pow, log
import numpy as np
import matplotlib.pyplot as plt


def main():
    min_iterations = 100
    eta = 0.1
    t = 200000

    x, y = get_x_y()
    w = np.random.uniform(-10, 10, x.shape[1])

    ein = []
    for i in range(t):
        if i < min_iterations:
            ein.append(insample_error(w, x, y))
        g = calculate_gradient(w, x, y)
        w = w - eta * g

        if np.sum(np.absolute(g)) < 0.000001 and i > min_iterations:
            break

    preds = get_predictions(w, x)

    plot_exp(ein, preds, x, min_iterations)


def get_predictions(w, x):
    preds = []
    for this_x in x:
        pred = round(logistic_fn(w, this_x))
        preds.append(pred)
    return [pred if pred > 0 else -1 for pred in preds]


def plot_exp(ein, preds, x, num_iterations):
    f, (ax1, ax2) = plt.subplots(1, 2)
    plt.tight_layout()

    # axes 1
    x_range = np.array(range(0, num_iterations))
    ax1.set(title='in-sample error',
            xlabel='iteration',
            ylabel='in-sample error'
            )
    ax1.plot(x_range, ein)

    # axes 2
    ax2.set(title='predictions',
            xlabel='x',
            ylabel='P(x = 1)'
            )
    ax2.scatter(x[:, 1], preds)
    plt.show()


def get_x_y():
    data = np.genfromtxt('training_data.csv', delimiter=',', dtype='int32')
    x, y = data[:, 0], data[:, 1]
    x = np.column_stack((np.ones_like(x), x))  # Let x0 equal 1
    return x, y


def calculate_gradient(w, x, y):
    gradient_sum = 0
    for this_x, this_y in zip(x, y):
        gradient_sum += partial_gradient(w, this_x, this_y)
    return - (gradient_sum / x.shape[0])


def partial_gradient(w, x, y):
    return (y * x) / (1 + pow(e, y * (np.dot(w, x))))


def logistic_fn(w, x):
    s = np.dot(w, x)
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
