# print(__doc__)

# # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# # License: BSD 3 clause

# # Standard scientific Python imports
# import matplotlib.pyplot as plt

# # Import datasets, classifiers and performance metrics
# from sklearn import datasets, svm, metrics
# from sklearn.model_selection import train_test_split

# digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)

# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)

# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False)

# # Learn the digits on the train subset
# clf.fit(X_train, y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title(f'Prediction: {prediction}')

# print(f"Classification report for classifier {clf}:\n"
#       f"{metrics.classification_report(y_test, predicted)}\n")
    
# disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

# from sklearn import datasets

# iris = datasets.load_iris()
# digits = datasets.load_digits
# print(digits.data)

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
# clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
# clf3 = GaussianNB()
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
# eclf1 = VotingClassifier(estimators=[
#         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
# eclf1 = eclf1.fit(X, y)
# print(eclf1.predict(X))

# np.array_equal(eclf1.named_estimators_.lr.predict(X),
#                eclf1.named_estimators_['lr'].predict(X))

# eclf2 = VotingClassifier(estimators=[
#         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#         voting='soft')
# eclf2 = eclf2.fit(X, y)
# print(eclf2.predict(X))

# eclf3 = VotingClassifier(estimators=[
#        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
#        voting='soft', weights=[2,1,1],
#        flatten_transform=True)
# eclf3 = eclf3.fit(X, y)
# print(eclf3.predict(X))

# print(eclf3.transform(X).shape)


# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from matplotlib import lines
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets import load_digits
# # iris = datasets.load_iris()
# # X_features = iris.data[:, 0:2] #Load first 2 features of data set
# # X_features2 = (iris.data[:, 3:]) #Load last feature of data set
# # X_features = np.concatenate((X_features, X_features2), axis=1) #Join this 2 above together
# # X_petal_len = iris.data[:, 2:3] #Load the 3rd feature
# # X_features_alldata = iris.data
# # y_label = iris.target
# # #y_label = y_label[1:3]

# iris = datasets.load_digits()
# X_features = iris.data[:, 0:2] #Load first 2 features of data set
# X_features2 = (iris.data[:, 3:]) #Load last feature of data set
# X_features = np.concatenate((X_features, X_features2), axis=1) #Join this 2 above together
# X_petal_len = iris.data[:, 2:3] #Load the 3rd feature
# X_features_alldata = iris.data
# y_label = iris.target
# #y_label = y_label[1:3]

# # Print feature names in data set
# #print(iris.feature_names[1:2, 2:])
# #print(iris.feature_names)

# # Print the target names in the data set
# #print(iris.target_names)
# #print(y_label)

# """ for eachFlower in range(len(iris.target)):
#     print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features[eachFlower]))
#     #print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features2[eachFlower]))
#     print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_petal_len[eachFlower]))
#     #To test that data was split accurately
#     print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features_alldata[eachFlower])) """

# #! Split the data, keep 1 third for testing
# X_trainfeatures, X_testfeatures, y_traininglabels, y_testlabels = train_test_split(X_features, y_label, test_size = .3, random_state = 7919)

# colors = ['r', 'g']

# #! Tree Training
# from sklearn import tree
# clfTree = tree.DecisionTreeClassifier()
# clfTree.fit(X_trainfeatures, y_traininglabels)
# #tree.plot_tree(clfTree.fit(X_trainfeatures, y_traininglabels))
# #clfTree_p = clfTree.predict(X_testfeatures)
# #plt.show()
# print("Accuracy for Decision Tree Classifier: " + str(clfTree.score(X_testfeatures, y_testlabels)))

# #? PLOTING
# # x = Sepal Length |  y = Sepal Width | z = Petal Width
# x_min, x_max = X_trainfeatures[:, 0].min(), X_trainfeatures[:, 0].max()
# y_min, y_max = X_trainfeatures[:, 1].min(), X_trainfeatures[:, 1].max()
# z_min, z_max = X_trainfeatures[:, 2].min(), X_trainfeatures[:, 2].max()


# # you need to define the min and max values from the data
# step_size = 0.05
# xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size),
#                          np.arange(y_min, y_max, step_size),
#                          np.arange(z_min, z_max, step_size))


# # the colors of the plot (parameter c)
# # should represent the predicted class value
# # we found this linewidth to work well
# clfTreePredict = clfTree.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
# c_pred = [colors[p-1] for p in clfTreePredict]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
# ax.set_ylabel(iris.feature_names[0])
# ax.set_xlabel(iris.feature_names[1])
# ax.set_zlabel(iris.feature_names[3])

# plt.title("Decision Tree Graph")

# # Setup Legend
# legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
# legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
# ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)

# plt.show()




# #! KNN
# from sklearn.neighbors import KNeighborsClassifier
# clfKNN = KNeighborsClassifier(n_neighbors=5)
# clfKNN.fit(X_trainfeatures, y_traininglabels)
# #clfKNN_p = clfKNN.predict(X_testfeatures)
# print("Accuracy for KNN Classifier: " + str(clfKNN.score(X_testfeatures, y_testlabels)))

# # the colors of the plot (parameter c)
# # should represent the predicted class value
# # we found this linewidth to work well
# clfKNNPredict = clfKNN.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
# c_pred = [colors[p-1] for p in clfKNNPredict]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
# ax.set_ylabel(iris.feature_names[0])
# ax.set_xlabel(iris.feature_names[1])
# ax.set_zlabel(iris.feature_names[3])

# plt.title("KNN Regression Graph")

# # Setup Legend
# legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
# legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
# ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)
# plt.show()


# # Load libraries
# from sklearn import datasets
# import matplotlib.pyplot as plt 
# # Load digits dataset
# digits = datasets.load_digits()

# # Create feature matrix
# X = digits.data

# # Create target vector
# y = digits.target

# # View the first observation's feature values
# X[0]

# # View the first observation's feature values as a matrix
# digits.images[0]

# # Visualize the first observation's feature values as an image
# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show()




# from sklearn import datasets
# import matplotlib.pyplot as plt 
 
# digits = datasets.load_digits()

# X = digits.data 
 
# y = digits.target 
 
# print(X[0])

# print(digits.images[0])

# from sklearn import datasets
# import matplotlib.pyplot as plt
 
# digits = datasets.load_digits()
 
# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show()

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
 
# digits = datasets.load_digits()
 
# X = digits.data
# y = digits.target
 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
 
# knn = KNeighborsClassifier(n_neighbors=7)
 
# knn.fit(X_train, y_train)
 
# print(knn.score(X_test, y_test))


# from sklearn.linear_model import LogisticRegression
 
# logisticRegr = LogisticRegression(solver='lbfgs', multi_class='auto')
# logisticRegr.fit(X_train, y_train)
# logisticRegr.predict(X_test[0].reshape(1,-1))
 
# logisticRegr.predict(X_test[0:10])
# predictions = logisticRegr.predict(X_test)
# score = logisticRegr.score(X_test, y_test)
# print(score)

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import metrics
 
# cm = metrics.confusion_matrix(y_test, predictions)
 
# plt.figure(figsize=(5,5))
# sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r');
# plt.ylabel('Actual label');
# plt.xlabel('Predicted label');
# all_sample_title = f'Accuracy Score: {score:.2f}'
# plt.title(all_sample_title, size = 12)
# plt.show()

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
 
# X = digits.data[:500]
# y = digits.target[:500]
 
# digits = datasets.load_digits()
# tsne = TSNE(n_components=2, random_state=0)
 
# X_2d = tsne.fit_transform(X)
 
# digits_ids = range(len(digits.target_names))
 
# plt.figure(figsize=(6, 5))
# colors = 'aqua', 'azure', 'coral', 'gold', 'green', 'fuchsia', 'maroon', 'purple', 'red', 'orange'
# for i, c, label in zip(digits_ids, colors, digits.target_names):
#     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
# plt.legend()
# plt.show()


# print(__doc__)

# # Code source: Gaël Varoquaux
# # Modified for documentation by Jaques Grobler
# # License: BSD 3 clause

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import datasets, cluster
# from sklearn.feature_extraction.image import grid_to_graph

# digits = datasets.load_digits()
# images = digits.images
# X = np.reshape(images, (len(images), -1))
# connectivity = grid_to_graph(*images[0].shape)

# agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
#                                      n_clusters=32)

# agglo.fit(X)
# X_reduced = agglo.transform(X)

# X_restored = agglo.inverse_transform(X_reduced)
# images_restored = np.reshape(X_restored, images.shape)
# plt.figure(1, figsize=(4, 3.5))
# plt.clf()
# plt.subplots_adjust(left=.01, right=.99, bottom=.01, top=.91)
# for i in range(4):
#     plt.subplot(3, 4, i + 1)
#     plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
#     if i == 1:
#         plt.title('Original data')
#     plt.subplot(3, 4, 4 + i + 1)
#     plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16,
#                interpolation='nearest')
#     if i == 1:
#         plt.title('Agglomerated data')
#     plt.xticks(())
#     plt.yticks(())

# plt.subplot(3, 4, 10)
# plt.imshow(np.reshape(agglo.labels_, images[0].shape),
#            interpolation='nearest', cmap=plt.cm.nipy_spectral)
# plt.xticks(())
# plt.yticks(())
# plt.title('Labels')
# plt.show()

# print(__doc__)


# # Code source: Gaël Varoquaux
# # Modified for documentation by Jaques Grobler
# # License: BSD 3 clause

# from sklearn import datasets

# import matplotlib.pyplot as plt

# #Load the digits dataset
# digits = datasets.load_digits()

# #Display the first digit
# plt.figure(1, figsize=(3, 3))
# plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

# print(__doc__)

# # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# # License: BSD 3 clause

# # Standard scientific Python imports
# import matplotlib.pyplot as plt

# # Import datasets, classifiers and performance metrics
# from sklearn import datasets, svm, metrics
# from sklearn.model_selection import train_test_split

# digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)

# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)

# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False)

# # Learn the digits on the train subset
# clf.fit(X_train, y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title(f'Prediction: {prediction}')


# print(f"Classification report for classifier {clf}:\n"
#       f"{metrics.classification_report(y_test, predicted)}\n")

import numpy
import numpy as np
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

classifiers = [
                (DummyClassifier(strategy="most_frequent"), "Simple Majority"),
                (DecisionTreeClassifier(criterion="entropy"), "Decision Tree"),
                (KNeighborsClassifier(n_neighbors=5), "5NN"),
                (KMeans(n_clusters=5), "5-means")
]

def read_data():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    return X, y

def plotting(X_traindata):
    colors = ['r','g']
    x_min, x_max = X_traindata[:, 0].min(), X_traindata[:, 0].max()
    y_min, y_max = X_traindata[:, 1].min(), X_traindata[:, 1].max()
    z_min, z_max = X_traindata[:, 2].min(), X_traindata[:, 2].max()
    step_size = 0.05
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size),
                                np.arange(y_min, y_max, step_size),
                                np.arange(z_min, z_max, step_size))
    clfTreePredict = clfTree.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    c_pred = [colors[p-1] for p in clfTreePredict]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
    ax.set_ylabel(iris.feature_names[0])
    ax.set_xlabel(iris.feature_names[1])
    ax.set_zlabel(iris.feature_names[3])


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

if __name__ == "__main__":
    X, y = read_data() # read training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    for clf, name in classifiers:
        clf.fit(X_train, y_train)
        newlabels = clf.predict(X_test)
        print("Classifier: ", name)
        print("Accuracy: ", metrics.accuracy_score(y_test, newlabels))
        print(metrics.confusion_matrix(y_test, newlabels), "\n")
        # if name == "Decision Tree":
        #     print(export_text(clf))
    plotting(X_train)
    nn1 = KNN(X, X[0],1)
    nn3 = KNN(X, X[0],3)
    nn5 = KNN(X, X[0],5)
    nn7 = KNN(X, X[0],7)
    nn9 = KNN(X, X[0],9)
    print("1-NN:")
    for k in nn1:
        print(k)

    print("3-NN:")
    for k in nn3:
        print(k)

    print("5-NN:")
    for k in nn5:
        print(k)

    print("7-NN:")
    for k in nn7:
        print(k)

    print("9-NN:")
    for k in nn9:
        print(k)
