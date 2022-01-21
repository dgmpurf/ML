text = 'python'
print(text, text[0], text[1], text[-1], text[-2])
print(text[0:-1])
print(text[0:6:2])

def main():
      print("hello world!")

if __name__ == "__main__":
  main()

"""
Machine Learning Lecture
January 25, 2021
"""

import numpy as np
import collections

# from google.colab import files

# def upload_data():
#   uploaded = files.upload()
"""
from google.colab import drive
drive.mount('/content/gdrive')


def read_data(filetype):
  if filetype == 'train':
    data = np.loadtxt(fname='/content/gdrive/My Drive/ML/lectures/traindata.csv', delimiter=',')
  else:
    data = np.loadtxt(fname='/content/gdrive/My Drive/ML/lectures/testdata.csv', delimiter=',')
  print("Your data", data)
  X = data[:,:-1]
  y = data[:,-1]
  return X, y
"""

def read_data(filetype):
  if filetype == 'train':
    data = np.loadtxt(fname='traindata.csv', delimiter=',')
  else:
    data = np.loadtxt(fname='testdata.csv', delimiter=',')
  print("Your data", data)
  X = data[:,:-1]
  y = data[:,-1]
  return X, y

def simple_majority_train(X, y):
  """ train
  """
  majority_class = collections.Counter(y).most_common(1)[0][0]
  print(majority_class)
  return majority_class

def simple_majority_test(X, y, majority_class):
  total = len(y)
  num_right = 0

  for i in range(total):
    label = majority_class
    if label == y[i]:
      num_right += 1
  print("total", total, "num_right", num_right, "accuracy", float(num_right)/float(total))

if __name__ == "__main__":
  # uploaded = upload_data()

  X, y = read_data('train') # read training data
  majority_class = simple_majority_train(X, y)
  X, y = read_data('test')
  simple_majority_test(X, y, majority_class)




import math

def entropy(pos, neg):
  """Compute entropy based on proportion of positive example (pos)
  and the porportion of negative example"""
  pf = pos / float(pos + neg)
  nf = neg / float(pos + neg)
  if pf ==0:
    term1 =0
  else:
    term1 = - pf * math.log(pf, 2.0) # -p log p
  if nf == 0:
    term2=0
  else:
    term2 = -nf * math.log(nf,2.0) # -q log q
  entropy = term1 + term2
  return entropy

def gain(pos, neg, splits):
  """Gain(S,A) = Entropy(S) - Sum_{c in A} |S_v|/|S| Entropy(S_v)
  """
  start = entropy(pos, neg)
  # print('start', start)
  result = start 
  for value in splits:
    size = (value[0]+value[1])/(pos + neg)
    child_entropy = entropy(value[0],value[1])
    print('value',value, size, child_entropy)
    result -= size * child_entropy
  return result
#example in our decition tree pdf, 14 total, 5 NO, 9 YES.
# print ('entropy', entropy(9, 5))
print ('entropy', entropy(2,2))
print ('ACE', gain(2, 2, [[1, 2],[1, 0]]))
print ('Ten', gain(2, 2, [[1, 1],[1, 1]]))
print ('FM', gain(2, 2, [[1, 1],[1, 1]]))
# # Outlook
# """
# If we choose Outlook 
# There are 3 Children in list, Sunny, Overcast, Rain. 
# So there are THREE in SPLITS list
# IN Sunny, 2 pos, 3 neg
# Overcast, 4 pos, 0 neg
# Rain, 3 pos 2 neg
# """
# print ('Outlook', gain(9, 5, [[2, 3],[4, 0],[3, 2]]))

# # Temp
# """
# Do the same thing with Outlook
# Hot, Mild, Cool. THREE Children
# """
# print ('Temp', gain(9, 5, [[2, 2],[4, 2],[3, 1]]))

# # Humidity
# """
# High, Normal. TWO Children
# """
# print ('Humidity', gain(9, 5, [[3, 4],[6, 1]]))

# # Wind
# """
# Weak, Strong. TWO Children
# """
# print ('Wind', gain(9, 5, [[6, 2],[3, 3]]))


"""
print (gain(3, 2, [[1,0],[2,2]]))

entropy 0.9709505944546686
start 0.9709505944546686
value [1, 0] 0.2 0.0
value [2, 2] 0.8 1.0
0.17095059445466854
"""
"""
print (gain(3, 2, [[3,0],[0,2]]))

entropy 0.9709505944546686
start 0.9709505944546686
value [3, 0] 0.6 0.0
value [0, 2] 0.4 0.0
0.9709505944546686
"""