import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import random


def train(X1, y, step_size, ):
    X_train = X1[:int(0.8 * len(X1))]
    y_train = y[:int(0.8 * len(y))]
    X_test = X1[int(0.8 * len(X1)):]
    y_test = y[int(0.8 * len(X1)):]
    w = np.zeros(len(X1[0]))
    theta = np.zeros(len(X1[0]))
    learning = 1
    epochs = 10
    wavg = np.zeros(len(X1[0]))
    for epoch in range(1, epochs + 1):
        w = (1 / step_size) * theta
        wavg[0] += w[0]
        wavg[1] += w[1]
        wavg[2] += w[2]
        i = random.randrange(0, len(X_train))
        if (y_train[i] * np.dot(X_train[i], w)) < 1:
            theta = theta + learning * (X_train[i] * y_train[i])
        else:
            theta = theta
    wavg[0] /= 10
    wavg[1] /= 10
    wavg[2] /= 10
    return wavg

def draw(X_train, y, wavg):
    for i in range(len(X_train)):
        if y[i] == 1:
            plt.scatter(X_train[i][1], X_train[i][2], s=50, marker='_', c='red')
        else:
            plt.scatter(X_train[i][1], X_train[i][2], s=50, marker='+', c='blue')

    x2 = [wavg[0], wavg[0], -wavg[2], wavg[0]]
    x3 = [wavg[0], wavg[0], wavg[2], -wavg[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=0.5, color='blue')

    x2 = [-wavg[0], -wavg[0], -wavg[2], wavg[0]]
    x3 = [-wavg[0], -wavg[0], wavg[2], -wavg[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')

    x2 = [-wavg[0], wavg[0], -wavg[2], wavg[0]]
    x3 = [wavg[0], -wavg[0], wavg[2], -wavg[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')


def test(X_test, y_test, wavg):
    for i in range(len(X_test)):
        if y_test[i] == 1:
            plt.scatter(X_test[i][1], X_test[i][2], s=50, marker='_', linewidths=2, c='red')
        else:
            plt.scatter(X_test[i][1], X_test[i][2], s=50, marker='+', linewidths=2, c='blue')

    x2 = [wavg[0], wavg[0], -wavg[2], wavg[0]]
    x3 = [wavg[0], wavg[0], wavg[2], -wavg[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')

    x2 = [-wavg[0], -wavg[0], -wavg[2], wavg[0]]
    x3 = [-wavg[0], -wavg[0], wavg[2], -wavg[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')

    x2 = [-wavg[0], wavg[0], -wavg[2], wavg[0]]
    x3 = [wavg[0], -wavg[0], wavg[2], -wavg[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale=1, color='blue')


def main():
    global data_dict
    X0, y = make_blobs(n_samples=100, n_features=2, centers=2,
                       cluster_std=1.05, random_state=10)

    X1 = np.c_[np.ones((X0.shape[0])), X0]  # add one to the x-values to incorporate bias


    X_train = X1[:int(0.8 * len(X1))]
    y_train = y[:int(0.8 * len(y))]
    X_test = X1[int(0.8 * len(X1)):]
    y_test = y[int(0.8 * len(X1)):]

    positive_x = []
    negative_x = []
    for i, label in enumerate(y):
        if label == 0:
            negative_x.append(X1[i])
        else:
            positive_x.append(X1[i])
        data_dict = {-1: np.array(negative_x), 1: np.array(positive_x)}

    max_fval = float()
    for y_i in data_dict:
        if np.amax(data_dict[y_i]) > max_fval:
            max_fval = np.amax(data_dict[y_i])

    print(max_fval)

    step_size = max_fval * 0.1
    weight = train(X1, y, step_size)
    draw(X_train,y,weight)
    test(X_test,y_test,weight)
