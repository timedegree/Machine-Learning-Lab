# %%
import numpy as np

# %%
a = np.array([1,2,3,4,5,6])

# %%
a[1]

# %%
a[0:2]

# %%
a[::2]

# %%
a[0] = 10
a

# %%
b = a[3:]
b 

# %%
b[0] = 40
a

# %%
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a

# %%
a[1, 3]
a

# %%
a.ndim

# %%
a.shape

# %%
a.size

# %%
a.dtype

# %%
np.zeros(2)

# %%
np.empty(2)

# %%
np.arange(4)

# %%
np.arange(2,9,2)

# %%
np.linspace(0,10,num=5)

# %%
arr = np.array([2,3,4,1,5,7,6])
np.sort(arr)

# %%
a = np.array([1,2,3])
b = np.array([4,5,6])
np.concatenate((a,b))

# %%
a = np.array([[1,2], [3,4]])
b = np.array([[5,6]])
np.concatenate((a,b),axis=0)

# %%
a = np.arange(6)
b = a.reshape(3,2)
b

# %%
np.reshape(a, (1, 6), order='C')

# %%
row_vector = a[np.newaxis, :]
row_vector.shape

# %%
col_vector = a[:, np.newaxis]
col_vector.shape

# %%
b = np.expand_dims(a, axis=1)
b.shape

# %%
c = np.expand_dims(a, axis=0)
c.shape

# %%
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[a > 5])

# %%
b = np.nonzero(a<5)
print(b)

# %%
a[b]

# %%
a1 = np.array([[1, 1],
           [2, 2]])
a2 = np.array([[3, 3],
           [4, 4]])
np.vstack((a1, a2))

# %%
np.hstack((a1, a2))

# %%
x = np.arange(1, 25).reshape(2, 12)
np.hsplit(x, 3)

# %%
np.hsplit(x, (3, 4))

# %%
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
data + ones

# %%
data - ones

# %%
data * data

# %%
data / data

# %%
a = np.array([1, 2, 3, 4])
a.sum()

# %%
b = np.array([[1, 1], [2, 2]])
b.sum(axis=0)

# %%
b.sum(axis=1)

# %%
data = np.array([1.0, 2.0])
data * 1.6


