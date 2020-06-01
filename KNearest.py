import numpy as np
from numpy import genfromtxt
from random import randrange
import sys

location = sys.argv[1]
k = int(sys.argv[2])

dataset = genfromtxt(location, delimiter=',')
dataset = dataset[1:-1]


class KNN:
    Euclidean_distance = 0
    label = -1
    features = []


training_percentage = 80
training_count = len(dataset) * (training_percentage / 100)
test_percentage = 100 - training_percentage
test_count = len(dataset) * (test_percentage / 100)

training_indexes = []
test_indexes = []
training_data = []
test_data = []

while len(test_indexes) < test_count:
    currentIndex = randrange(len(dataset))
    if currentIndex not in test_indexes:
        test_indexes.append(currentIndex)
for currentIndex in test_indexes:
    test_data.append(dataset[currentIndex])

for i in range(0, len(dataset)):
    if i not in test_indexes:
        training_indexes.append(i)
        training_data.append(dataset[i])

correctPredictionCount = 0

for Row in test_data:
    Neighbors = []
    Test_Actual_Label = Row[-1]
    Feature = Row[0:-1]
    for Training in training_data:
        TrainingFeature = Training[0:-1]
        TrainingLabel = Training[-1]
        dist = np.linalg.norm(np.asarray(Feature) - np.asarray(TrainingFeature))
        knn = KNN()
        knn.DistanceFromPoint = dist
        knn.Label = TrainingLabel
        knn.Feature = TrainingFeature
        Neighbors.append(knn)
    Neighbors.sort(key=lambda x: x.DistanceFromPoint, reverse=False)
    KNearestNeighbors = Neighbors[0:k]

    Positive = 0
    Negative = 0
    for knn in KNearestNeighbors:
        if knn.Label == 1.0:
            Positive = Positive + 1
        else:
            Negative = Negative + 1
    Test_Predicted_Label = 0
    if Positive > Negative:
        Test_Predicted_Label = 1.0
    else:
        Test_Predicted_Label = 0.0
    if Test_Predicted_Label == Test_Actual_Label:
        correctPredictionCount += 1

print("Accuracy of the model is:")
print(correctPredictionCount / len(test_data) * 100)
