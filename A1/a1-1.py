import math
import numpy as np
import collections
import random

def read_data():
  data = np.loadtxt(fname='./alldata.csv', delimiter=',')
#   print("shape of data:",data.shape)
#   print("datatype of data:",data.dtype)
#   print(data)
  sample_data = []
  sorted_sample_data = []
  training_data = []
  sorted_training_data = []
  for i in range(500):
    sample_index = random.randint(1,1000)
    sorted_sample_data = np.append([], sorted(data[sample_index]), axis=0)
    training_data = np.delete(data, sample_index, 0)
  for i in training_data:
    sorted_training_data = np.append([],sorted(i), axis=0)
  
  print("sample",sorted_sample_data)
  print("traning", sorted_training_data)
  X = sorted_training_data[:,:-1]
  y = sorted_training_data[:,-1] 
  return X, y
if __name__ == "__main__":
    read_data()

# ints = [1,2,3]
# string_ints = [str(int) for int in ints]
# str_of_ints = ",".join(string_ints)

# print(str_of_ints)
