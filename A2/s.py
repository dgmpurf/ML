import numpy
import collections
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import numpy as np

classifiers = [
                (DummyClassifier(strategy="most_frequent"), "Simple Majority"),
                (DecisionTreeClassifier(criterion="entropy"), "Decision Tree"),
                (KNeighborsClassifier(n_neighbors=5), "5NN"),
                (KMeans(n_clusters=5), "5-means")
]

def read_data():
    digits = datasets.load_digits()
    X_1 = digits.data[:,:2]
    X_2 = digits.data[:,3:]
    X = numpy.concatenate((X_1, X_2), axis=1)
    y = digits.target
    return X, y

def KNN_distance(pre,nex):
    distance = 0.0
    for i in range(len(pre)-1):
        distance += pow((pre[i]-nex[i]),2)
    return sqrt(distance)

def KNN(traindata, testdata, n):
    distance = list()
    for i in traindata:
        dist = KNN_distance(testdata, i)
        distance.append((i,dist))
    distance.sort(key=lambda tup:tup[1])
    neighbors = list()
    for j in range(n):
        neighbors.append(distance[j][0])
    return neighbors

def predict(traindata, testdata, n):
    neighbors = KNN(traindata, testdata, n)
    outvalues = [row[-1] for row in neighbors]
    preciction = max(set(outvalues), key=outvalues.count)
    return preciction

def accuracy_metric(actual, predicted):
    	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

if __name__ == "__main__":
    X, y = read_data() # read training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    nn1 = predict(X, X[0],9)
    nn3 = predict(X, X[0],3)
    nn5 = predict(X, X[0],5)
    nn7 = predict(X, X[0],7)
    nn9 = predict(X, X[0],9)
    print('Expected %d, Got %d.' % (X[0][-1], nn1))
    # print("Classifier: 1NN Accuracy: ", metrics.accuracy_score(X[0][-1], nn1),"\n")
    clfKNN1 = KNeighborsClassifier(n_neighbors=5)
    clfKNN1.fit(X, y)
    clfKNN3 = KNeighborsClassifier(n_neighbors=5)
    clfKNN3.fit(X, y)
    clfKNN5 = KNeighborsClassifier(n_neighbors=5)
    clfKNN5.fit(X, y)
    clfKNN7 = KNeighborsClassifier(n_neighbors=5)
    clfKNN7.fit(X, y)
    clfKNN9 = KNeighborsClassifier(n_neighbors=5)
    clfKNN9.fit(X, y)
    print("Classifier: 1NN Accuracy: " + str(clfKNN1.score(X_test, y_test)))
    print("Classifier: 3NN Accuracy: " + str(clfKNN3.score(X_test, y_test)))
    print("Classifier: 5NN Accuracy: " + str(clfKNN5.score(X_test, y_test)))
    print("Classifier: 7NN Accuracy: " + str(clfKNN7.score(X_test, y_test)))
    print("Classifier: 9NN Accuracy: " + str(clfKNN9.score(X_test, y_test)))

    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    # print("3-NN:")
    # for k in nn3:
    #     print(k)

    # print("5-NN:")
    # for k in nn5:
    #     print(k)

    # print("7-NN:")
    # for k in nn7:
    #     print(k)

    # print("9-NN:")
    # for k in nn9:
    #     print(k)