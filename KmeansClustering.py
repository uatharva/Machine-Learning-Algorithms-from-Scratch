import numpy as np
from numpy import genfromtxt
from random import randrange
import sys

location = sys.argv[1]
distanceMetric = sys.argv[2]
k = sys.argv[3]

my_data = genfromtxt(location, delimiter=',')
my_data = my_data[1:-1]
row_count = len(my_data)

class Clusters:
    Centroid = []
    Points = []
    Labels = -1


Cluster_list = []
Cluster_centers = []


def KMeansClustering():

    for i in range(int(k)):
        current_cluster_center = randrange(row_count)
        if current_cluster_center not in Cluster_centers:
            Cluster_centers.append(current_cluster_center)
            current_index = Clusters()
            current_index.PointIndexes = []
            current_index.Centroid = my_data[current_cluster_center][0:-1]
            Cluster_list.append(current_index)

    for i in range(row_count):
        minDistance = 10000
        nearest_cluster = Cluster_list[0]
        for current_cluster in Cluster_list:

            if distanceMetric == 'Manhattan':
                distance = sum(abs(a - b) for a, b in zip(current_cluster.Centroid, my_data[i][0:-1]))

            else:
                distance = np.linalg.norm(np.asarray(current_cluster.Centroid) - np.asarray(my_data[i][0:-1]))

            if distance < minDistance:
                nearest_cluster = current_cluster
                minDistance = distance
        nearest_cluster.PointIndexes.append(i)

    temp = False
    while not temp:
        for current_cluster in Cluster_list:
            points_sum = [0, 0, 0, 0, 0]
            for curPointIndex in current_cluster.PointIndexes:
                points_sum = points_sum + my_data[curPointIndex][0:-1]
            points_sum = points_sum / len(current_cluster.PointIndexes)
            current_cluster.Centroid = points_sum

        temp = True
        for i in range(row_count):
            nearest_cluster = Cluster_list[0]
            min_distance = 10000
            for current_cluster in Cluster_list:
                if distanceMetric == 'Manhattan':
                    curDistance = sum(abs(a - b) for a, b in zip(current_cluster.Centroid, my_data[i][0:-1]))
                else:
                    curDistance = np.linalg.norm(np.asarray(current_cluster.Centroid) - np.asarray(my_data[i][0:-1]))
                if curDistance < min_distance:
                    nearest_cluster = current_cluster
                    min_distance = curDistance
            if i not in nearest_cluster.PointIndexes:
                temp = False
                for curCluster in Cluster_list:
                    if i in curCluster.PointIndexes:
                        curCluster.PointIndexes.remove(i)

                nearest_cluster.PointIndexes.append(i)


def Output():
    i = 1
    print()
    global Cluster_list
    for current_cluster in Cluster_list:
        number_of_ones = 0
        number_of_zeroes = 0
        for current_index in current_cluster.PointIndexes:
            if my_data[current_index][-1] == 1.0:
                number_of_ones = number_of_ones + 1
            else:
                number_of_zeroes = number_of_zeroes + 1
        Ratio_of_zeroes = number_of_zeroes / len(current_cluster.PointIndexes)
        Ratio_of_ones = number_of_ones / len(current_cluster.PointIndexes)

        if Ratio_of_ones > Ratio_of_zeroes:
            current_cluster.Label = 1.0
        else:
            current_cluster.Label = 0.0
        print("Cluster " + str(i) + " details:")
        print("Positive diagnosis count: " + str(number_of_ones))
        print("Negative diagnosis count: " + str(number_of_zeroes))
        print("Total diagnosis count : " + str(len(current_cluster.PointIndexes)))
        print("Percentage of positive diagnosis " + str(Ratio_of_ones * 100) + "%")
        print("Predicted label of cluster: " + str(current_cluster.Label))
        print()
        i = i + 1


KMeansClustering()
Output()
