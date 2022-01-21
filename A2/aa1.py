# import numpy
# import collections
# from sklearn.dummy import DummyClassifier

# def read_data(type):
#   if type == 'train':
#     data = numpy.loadtxt(fname='train.csv', delimiter=',')
#   else:
#     data = numpy.loadtxt(fname='test.csv', delimiter=',')
#   X = data[:,:-1]
#   y = data[:,-1]
#   return X, y

# def simple_majority_train(X, y):
#   majority_class = collections.Counter(y).most_common(1)[0][0]
#   print(majority_class)
#   return majority_class

# def simple_majority_test(X, y, majority_class):
#   total = len(y)
#   true_positive = 0
#   false_positive = 0
#   true_negative = 0
#   false_negative = 0

#   for i in range(total):
#     label = majority_class
#     if label == y[i]:
#         if y[i] == 0.0:
#             true_negative += 1
#         else:
#             false_negative += 1
#     else: 
#         if y[i] == 0.0:
#             false_positive += 1
#         else:
#             true_positive += 1
#     report_statistics(total, true_positive,false_positive,
#                           true_negative,false_negative)

# def sklearn_majority_train(X,y):
#     classifier = DummyClassifier(strategy="most_frequent")
#     classifier.fit(X, y)
#     return classifier

# def sklearn_majority_test(classifier, X, y):
#     newlabels=classifier.predict(X)
#     correct = 0
#     total = len(y)
#     for i in range(total):
#         if newlabels[i] == y[i]:
#             correct += 1
#     print("sklearn majority accuracy is", float(correct)/float(total))

# def report_statistics(total, tp, fp, tn, fn):
#     print("total", total, "tp", tp, "fp", fp, "tn", tn, "fn", fn)
#     print("simple majority accuracy is", float(tp+tn)/float(total))

# if __name__ == "__main__":
#   # uploaded = upload_data()

#   X, y = read_data('train') # read training data
#   majority_class = simple_majority_train(X, y)
#   classifier = sklearn_majority_train(X,y)
#   X, y = read_data('test')
#   simple_majority_test(X, y, majority_class)
#   sklearn_majority_test(classifier, X, y)

################## Majority, Split, SKlearn
# import numpy
# import collections
# from sklearn.dummy import DummyClassifier
# from sklearn import metrics
# from sklearn.model_selection import train_test_split

# def read_data():
#     data = numpy.loadtxt(fname='alldata.csv', delimiter=',')

#     X = data[:,:-1]
#     y = data[:,-1]
#     return X, y

# def sklearn_majority_train(X,y):
#     classifier = DummyClassifier(strategy="most_frequent")
#     classifier.fit(X, y)
#     return classifier

# def sklearn_majority_test(classifier, X, y):
#     newlabels = classifier.predict(X)
#     print("accuracy is ", metrics.accuracy_score(y, newlabels))
#     print(metrics.confusion_matrix(y, newlabels))

# if __name__ == "__main__":
#     X, y = read_data() # read training data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#     classifier = sklearn_majority_train(X_train,y_train)
#     X, y = read_data()
#     sklearn_majority_test(classifier, X_test, y_test)


############## Decision tree, KNN, Kmean
import numpy
import collections
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

classifiers = [
                (DummyClassifier(strategy="most_frequent"), "Simple Majority"),
                (DecisionTreeClassifier(criterion="entropy"), "Decision Tree"),
                (KNeighborsClassifier(n_neighbors=5), "5NN"),
                (KMeans(n_clusters=5), "5-means")
]

def read_data():
    data = numpy.loadtxt(fname='alldata.csv', delimiter=',')

    X = data[:,:-1]
    y = data[:,-1]
    return X, y

if __name__ == "__main__":
    X, y = read_data() # read training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    for clf, name in classifiers:
        clf.fit(X_train, y_train)
        newlabels = clf.predict(X_test)
        print("Classifier: ", name)
        print("Accuracy: ", metrics.accuracy_score(y_test, newlabels))
        print(metrics.confusion_matrix(y_test, newlabels), "\n")
        if name == "Decision Tree":
            print(export_text(clf))