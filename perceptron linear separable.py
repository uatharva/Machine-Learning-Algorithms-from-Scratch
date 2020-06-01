import numpy as np
from numpy import genfromtxt


class perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def prediction(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if type(summation) == type(np.array([1,1])):
            summation = sum(summation)
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def training(self, train_input, label):
        for _ in range(self.threshold):
            for inputs, label in zip(train_input, labels):
                prediction = self.prediction(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


training_inputs = []
labels = []
data = genfromtxt('linearly-separable-dataset.csv', delimiter=',')
for row in data:
    if not (row[0] > 0):
        continue
    final = row[0:-1]
    training_inputs.append(np.array(final))
    labels.append(row[2])

labels = np.array(labels)

perceptron = perceptron(2)
perceptron.training(training_inputs, labels)

correct = 0
total = 0
for Row in data:
    npa = np.asarray(Row[0:-1])
    if np.add(np.dot(npa, perceptron.weights[1:]), perceptron.weights[0]) > 0:
        if Row[-1] == 1.0:
            correct += 1
            total += 1
        else:
            total += 1
    elif np.dot(npa, perceptron.weights[1:]) < 0:
        if Row[-1] == 0.0:
            correct += 1
            total += 1
        else:
            total += 1
print("ERM - ",correct / total)

folds = []
foldlables = []
data1 = data[:, 2]
datax = data[:,:]
data = data[:, 0:2]
x = int(data.shape[0] / 10)
old = 0
for i in range(1, 10):
    folds.append(data[old:old + x])
    for i in data1[old:old + x]:
        foldlables.append(i)
    old = old + x

folds[0] = folds[0][1:]
foldlables = foldlables[1:]
nn=0
for i in range(0, 10):
    y = []
    temp = folds[0:i] + folds[i + 1:]
    for xx in temp:
        for j in xx:
            y.append(j)
    z= foldlables[0:int(i*x)] + foldlables[int((i*x)+x):]
    perceptron.training(y,z)

    correct = 0
    total = 0
    for Row in datax[i*x:i*x+x]:
        npa = np.asarray(Row[0:-1])
        if np.add(np.dot(npa, perceptron.weights[1:]), perceptron.weights[0]) > 0:
            if Row[-1] == 1.0:
                correct += 1
                total += 1
            else:
                total += 1
        elif np.dot(npa, perceptron.weights[1:]) < 0:
            if Row[-1] == 0.0:
                correct += 1
                total += 1
            else:
                total += 1
    nn += correct / total
print("10 Fold Cross Validation - ",nn/10)