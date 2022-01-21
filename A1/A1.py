# Decision tree learning
#
# Assumes discrete features. Examples may be inconsistent. Stopping condition for tree
# generation is when all examples have the same class, or there are no more features
# to split on (in which case, use the majority class). If a split yields no examples
# for a particular feature value, then the classification is based on the parent's
# majority class.

import math
import numpy as np
import collections
import random

def read_data():
  data = np.loadtxt(fname='./alldata.csv', delimiter=',')
  sorted_sample_data = []
  training_data = []
  sorted_training_data = []
  for i in range(100):
    sample_index = random.randint(1,1000)
    sorted_sample_data = np.append([], sorted(data[sample_index]), axis=0)
    training_data = np.delete(data, sample_index, 0)
  for i in training_data:
    sorted_training_data = np.append([],sorted(i), axis=0)

  print("sample",sorted_sample_data)
  print("traning", sorted_training_data)
#   if filetype == 'train':
#     outP_data = sorted_training_data
#   else:
#     outP_data = sorted_sample_data

#   return outP_data


if __name__ == "__main__":
   read_data()

#    test_instance = read_data("test")
#    test_class = classify(tree, test_instance)
#    print("\nTest instance: " + str(test_instance))
#    print("  class = " + test_class)