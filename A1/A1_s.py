import copy
import math
import pandas as pd
import numpy as np
import random
from random import sample
import matplotlib
import matplotlib.pyplot as plt

# global variables
num_bins = 2
continuous = True
num_trials = 10

class TreeNode:
    def __init__(self, majClass):
        self.split_feature = -1 # -1 indicates leaf node
        self.children = {} # dictionary of {feature_value: child_tree_node}
        self.majority_class = majClass
        
def build_tree(examples):
    if not examples:
        return None
    # collect sets of values for each feature index, based on the examples
    features = {}
    for feature_index in range(len(examples[0]) - 1):
        features[feature_index] = set([example[feature_index] for example in examples])
    return build_tree_1(examples, features)
    
def build_tree_1(examples, features):
    tree_node = TreeNode(majority_class(examples))
    # if examples all have same class, then return leaf node predicting this class
    if same_class(examples):
        return tree_node
    # if no more features to split on, then return leaf node predicting majority class
    if not features:
        return tree_node
    # split on best feature and recursively generate children
    best_feature_index = best_feature(features, examples)
    tree_node.split_feature = best_feature_index
    remaining_features = features.copy()
    remaining_features.pop(best_feature_index)
    for feature_value in features[best_feature_index]:
        split_examples = filter_examples(examples, best_feature_index, feature_value)
        tree_node.children[feature_value] = build_tree_1(split_examples, remaining_features)
    return tree_node

def majority_class(examples):
    classes = [example[-1] for example in examples]
    return max(set(classes), key = classes.count)

def same_class(examples):
    classes = [example[-1] for example in examples]
    return (len(set(classes)) == 1)

def best_feature(features, examples):
    # Return index of feature with lowest entropy after split
    best_feature_index = -1
    best_entropy = 2.0 # max entropy = 1.0
    for feature_index in features:
        se = split_entropy(feature_index, features, examples)
        if se < best_entropy:
            best_entropy = se
            best_feature_index = feature_index
    return best_feature_index

def split_entropy(feature_index, features, examples):
    # Return weighted sum of entropy of each subset of examples by feature value.
    se = 0.0
    for feature_value in features[feature_index]:
        split_examples = filter_examples(examples, feature_index, feature_value)
        se += (float(len(split_examples)) / float(len(examples))) * entropy(split_examples)
    return se

def entropy(examples):
    classes = [example[-1] for example in examples]
    classes_set = set(classes)
    class_counts = [classes.count(c) for c in classes_set]
    e = 0.0
    class_sum = sum(class_counts)
    for class_count in class_counts:
        if class_count > 0:
            class_frac = float(class_count) / float(class_sum)
            e += (-1.0)* class_frac * math.log(class_frac, 2.0)
    return e

def filter_examples(examples, feature_index, feature_value):
    # Return subset of examples with given value for given feature index.
    return list(filter(lambda example: example[feature_index] == feature_value, examples))

def print_tree(tree_node, feature_names, depth = 1):
    indent_space = depth * "  "
    if tree_node.split_feature == -1: # leaf node
        print(indent_space + feature_names[-1] + ": " + tree_node.majority_class)
    else:
        for feature_value in tree_node.children:
            print(indent_space + feature_names[tree_node.split_feature] + " == " + feature_value)
            child_node = tree_node.children[feature_value]
            if child_node:
                print_tree(child_node, feature_names, depth+1)
            else:
                # no child node for this value, so use majority class of parent (tree_node)
                print(indent_space + "  " + feature_names[-1] + ": " + tree_node.majority_class)

def classify(tree_node, instance):
    if tree_node.split_feature == -1:
        return tree_node.majority_class
    child_node = tree_node.children[instance[tree_node.split_feature]]
    if child_node:
        return classify(child_node, instance)
    else:
        return tree_node.majority_class

def discretize(data):
  n1 = len(data)
  n2 = n1 / 2
  values = copy.copy(data)
  values.sort()
  threshold = values[n2]
  newdata = np.zeros(n1, dtype=str)
  for i in range(n1):
    if data[i] <= threshold:
      newdata[i] = 'small'
    else:
      newdata[i] = 'large'
  return newdata

def plot_results(x, y):
  fig, ax = plt.subplots()
  ax.plot(x, y)
  ax.set(xlabel='training size', ylabel='classification accuracy',
         title='Learning curve')
  ax.grid()
  plt.show()


if __name__ == "__main__":
   random.seed()
   df = pd.read_csv('./alldata.csv')
   data = df.to_numpy()
   d = len(data[0])
   n = len(data)
   if continuous:
      examples = np.zeros(np.shape(data), dtype=str)
      for i in range(d-1):
         examples[:,i] = discretize(data[:,i])
   else:
      examples = data
   for i in range(n):
     examples[i][d-1] = str(data[i][d-1])
   """
   feature_names = ["Color", "Type", "Origin", "Stolen"]
   
   examples = [
       ["Red", "Sports", "Domestic", "Yes"],
       ["Red", "Sports", "Domestic", "No"],
       ["Red", "Sports", "Domestic", "Yes"],
       ["Yellow", "Sports", "Domestic", "No"],
       ["Yellow", "Sports", "Imported", "Yes"],
       ["Yellow", "SUV", "Imported", "No"],
       ["Yellow", "SUV", "Imported", "Yes"],
       ["Yellow", "SUV", "Domestic", "No"],
       ["Red", "SUV", "Imported", "No"],
       ["Red", "Sports", "Imported", "Yes"]
       ]
   tree = build_tree(examples)
   print("Tree:")
   print_tree(tree, feature_names)
   test_instance = ["Red", "SUV", "Domestic"]
   test_class = classify(tree, test_instance)
   print("\nTest instance: " + str(test_instance))
   print("  class = " + test_class)
   """
   samples = []
   results = []
   for trials in range(num_trials):
      count = 0
      for i in range(1, 200, 5):
         test_set = sample(examples, 100)
         train_set = sample(examples, i)
         tree = build_tree(train_set)
         right = 0
         for j in range(100):
            test_class = classify(tree, test_set[j])
            if test_class == test_set[j][d-1]:
               right += 1
         if trials == 0:
            samples.append(i)
            results.append(right)
         else:
            results[count] += right
         count += 1
   for i in range(len(results)):
      results[i] = float(results[i]) / float(num_trials)
   plot_results(samples, results)