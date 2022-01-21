import numpy as np
from numpy.linalg import matrix_power
# a = np.array([1,2,3], dtype='int32')
# print(a)


# b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
# print(b)

# # Get Dimension
# a.ndim
# print(a.ndim)
# # Get Shape
# b.shape
# print(b.shape)
# # Get Type
# a.dtype
# print(a.dtype)
# # Get Size
# a.itemsize
# print(a.itemsize)
# # Get total size
# a.nbytes
# print(a.nbytes)
# # Get number of elements
# a.size
# print(a.size)

# a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
# print(a)

# # Get a specific element [r, c]
# print(a[1, 5])

# # Get a specific row
# print(a[0, :])

# # Get a specific column
# print(a[:, 2])

# # Getting a little more fancy [startindex:endindex:stepsize]
# print(a[0, 1:-1:2])
# print('''################\n''')

# a[1,5] = 20
# print(a)
# a[:,2] = [1,2]
# print(a)

# print('3D examples')
# #3d, 0 1 1 3层[]
# b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(b)

# # Get specific element (work outside in)
# print(b[0,1,1])

# #这是一层一层的，b[最外层，第二层, 第三层]
# print(b[1,:,:])
# print(b[:,1,:])
# print(b[:,:,1])

# # replace 
# b[:,1,:] = [[9,9],[8,8]]
# print(b , '\n\n')


# # All 0s matrix
# s = np.zeros((2,3))
# print(s)

# # All 1s matrix
# s = np.ones((4,2,2), dtype='int32')
# print(s)

# # Any other number
# s = np.full((2,2), 99)
# print(s)

# # Any other number (full_like)
# s = np.full_like(a, 4)
# print(s)

# # Random decimal numbers
# s = np.random.rand(4,3)
# print(s)

# # Random Integer values
# s = np.random.randint(-4,8, size=(4,3))
# print(s)

# # The identity matrix
# s = np.identity(5) #可以添加 , dtype='int32'
# print(s,'\n\n\n')

# # Repeat an array
# arr = np.array([[1,2,3]])
# r1 = np.repeat(arr,3, axis=0)
# print(r1)


# output = np.ones((5,5))
# print(output)

# z = np.zeros((3,3))
# z[1,1] = 9
# print(z)

# output[1:-1,1:-1] = z
# print(output)

# a = np.array([1,2,3,4,5,6,7,8,9])
# b = a.copy()
# b[4] = 100

# print(b)

# a = np.array([1,2,3,4])
# print(a)

# s = a + 2
# print(s)

# c = a - 2
# print(c)

# c = a * 2

# c = a / 2

# b = np.array([1,0,1,0])
# s = a + b
# print(s)

# s = a ** 2
# print(s)

# np.cos(a)

# more in https://numpy.org/doc/stable/reference/routines.math.html

# a = np.ones((2,3))
# print(a)

# b = np.full((3,2), 2)
# print(b)

# s = np.matmul(a,b)
# print(s)


# Find the determinant
# c = np.identity(10)
# s = np.linalg.det(c)
# print(s)

# a = np.random.randint(0,10,size=(2,3))
# print(a)
# b = np.random.randint(0,10,size=(3,2))
# print(b)
# s = np.matmul(a,b)
# print(s)

## Reference docs (https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)

# Determinant
# Trace
# Singular Vector Decomposition
# Eigenvalues
# Matrix Norm
# Inverse
# Etc...
# i = np.array([[1, 2], [4, 5]])
# i = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# s = matrix_power(i, 2)
# print(s)

# stats = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
# print(stats)

# np.min(stats)
# # 0 col, 1 row
# s = np.max(stats, axis=1)
# print(s)

# s=np.sum(stats, axis=0)
# print(s)
# s=np.sum(stats, axis=1)
# print(s)

# reshape
# before = np.array([[1,2,3,4],[5,6,7,8]])
# print(before)

# after = before.reshape((4,2))
# after = before.reshape((1,8))
# after = before.reshape((8,1))
# print(after)

# Vertically stacking vectors
# v1 = np.array([1,2,3,4])
# v2 = np.array([5,6,7,8])

# s = np.vstack([v1,v2,v1,v2])
# print(s)

# Horizontal  stack
# h1 = np.ones((2,4)) # 可以加,dtype='int32'
# h2 = np.zeros((2,2))
# print(h1)
# print(h2)
# s = np.hstack((h1,h2))#不可以加,dtype='int32'
# print(s)

# filedata = np.genfromtxt('data.txt', delimiter=',')
# filedata = filedata.astype('int32')
# print(filedata)

# s=(~((filedata > 12) & (filedata < 40)))
# print(s)