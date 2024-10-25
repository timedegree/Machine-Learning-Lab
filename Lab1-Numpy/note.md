# Numpy

## 导入

~~~py
import numpy as np
~~~

## Array 基础

- `np.array()` 创建数组，如`a = np.array([1,2,3,4,5,6])`

- array 也可进行切片访问元素

~~~py
  a[0]
  # 1
  a[0:2]
  # array([1, 2])
  a[::2]
  # array([1, 3, 5])
~~~

- 通过访问元素并修改，如

~~~python
a[0] = 10
a
#array([10,  2,  3,  4,  5,  6])
~~~

- 通过使用高维列表作为参数，可创建多维数组

~~~py
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a
#array([[ 1,  2,  3,  4],
#       [ 5,  6,  7,  8],
#       [ 9, 10, 11, 12]])
~~~

- 多维数组通过`a[row,col]`的方式访问元素

~~~python
a[1, 3]
a
# 8
~~~

> [!IMPORTANT]
>
> 需要注意的是，将数组的某个切片赋值给一个新的变量名时，所进行的拷贝为浅拷贝，关注以下例子：
>
> ~~~python
> b = a[3:]
> b 
> # array([4, 5, 6])
> b[0] = 40
> a
> # array([10, 2, 3, 40, 5, 6])
> ~~~

## Array 属性

- 通过 `ndim` 获取数组的维度，如 `a.ndim` 值为 `2`
- 通过 `shape` 获取数组各维度元素个数，如 `a.shape` 值为 `(3,4)`
- `size` 记录了数组中总的元素个数，如 `a.size` 值为 `12`
- `dtype` 记录数组中元素类型，如 `a.dtype` 返回 `dtype('int64')`

## 快速创建常用 Array

- `np.zeros()` 创建零数组

~~~python
np.zeros(2)
# array([0., 0.])
~~~

- `np.ones()` 创建元素全为 `1` 的数组

~~~python
np.ones(2)
# array([1., 1.])
~~~

- `np.empty()` 根据内存状态随机对数组进行赋值

~~~python
np.empty(2)
# array([114514., 1919810.])
~~~

- `np.arange()` 创建一个 `range` 的数组（默认从 `0` 开始）

~~~python
np.arange(4)
# array([0, 1, 2, 3])
np.arrange(2,9,2)
# array([2, 4, 6, 8])
~~~

- `np.linspace()` 根据等差数列创建数组

~~~python
np.linspace(0, 10, num=5)
# array([ 0. ,  2.5,  5. ,  7.5, 10. ])
~~~

> [!NOTE]
>
> 部分函数默认生成的数据类型为 `np.float64`，可通过更改 `dtype` 参数修改生成的数据类型
>
> ~~~python
> np.ones(2, dtype=np.int64)
> # array([1,1])
> ~~~

## Array 排序与拼接

- `np.sort()` 对数组进行排序

~~~python
arr = np.array([2,3,4,1,5,7,6])
np.sort(arr)
# array([1, 2, 3, 4, 5, 6, 7])
~~~

- `np.concatenate()` 对数组进行拼接

~~~python
a = np.array([1,2,3])
b = np.array([4,5,6])
np.concatenate((a,b))
# array([1, 2, 3, 4, 5, 6])

a = np.array([[1,2], [3,4]])
b = np.array([[5,6]])
np.concatenate((a,b),axis=0)
# array([[1, 2],
#      [3, 4],
#      [5, 6]])
~~~

## 重塑 Array

- `arr.reshape()` 对数组形状进行重组

~~~python
a = np.arange(6)
b = a.reshape(3,2)
b
# array([[0, 1],
#      [2, 3],
#      [4, 5]])
~~~

- 或使用 `np.reshape` 指定一些可选参数

~~~python
np.reshape(a, shape=(1, 6), order='C')
# array([[0, 1, 2, 3, 4, 5]])
~~~

## 将一维数组转化为二维

- `np.newaxis` 添加新轴

~~~python
row_vector = a[np.newaxis, :]
row_vector.shape
# (1, 6)

col_vector = a[:, np.newaxis]
col_vector.shape
# (6, 1)
~~~

- `np.expand_dims` 在指定位置插入新轴

~~~python
b = np.expand_dims(a, axis=1)
b.shape
# (6, 1)

c = np.expand_dims(a, axis=0)
c.shape
# (1, 6)
~~~

## 索引与切片

- numpy 可使用 `arr[bool expression]` 的方式来方便的筛选元素

~~~python
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[a > 5])
# [ 6  7  8  9 10 11 12]
~~~

- 也可使用 `np.nonzero()` 对非零元素进行筛查,给出的结果是二元组，一个代表行坐标一个代表列坐标,此时可通过该元组提取数组中被筛选出的内容

~~~python
b = np.nonzero(a<5)
print(b)
# (array([0, 0, 0, 0], dtype=int64), array([0, 1, 2, 3], dtype=int64))
print(a[b])
# array([1, 2, 3, 4])
~~~

## 堆叠 拆分 复制

-  `np.vstack` 和 `np.hstack` 可分别进行竖直堆叠和水平堆叠

~~~python
a1 = np.array([[1, 1],
           [2, 2]])
a2 = np.array([[3, 3],
           [4, 4]])
np.vstack((a1, a2))
# array([[1, 1],
#      [2, 2],
#      [3, 3],
#      [4, 4]])
np.hstack((a1, a2))
# array([[1, 1, 3, 3],
#      [2, 2, 4, 4]])
~~~

- 使用 `np.hsplit()` 进行拆分

~~~python
x = np.arange(1, 25).reshape(2, 12)
np.hsplit(x, 3)
# [array([[ 1,  2,  3,  4],
#      [13, 14, 15, 16]]),
# array([[ 5,  6,  7,  8],
#      [17, 18, 19, 20]]),
# array([[ 9, 10, 11, 12],
#      [21, 22, 23, 24]])]

np.hsplit(x, (3, 4))
# [array([[ 1,  2,  3],
#      [13, 14, 15]]),
# array([[ 4],
#      [16]]),
# array([[ 5,  6,  7,  8,  9, 10, 11, 12],
#      [17, 18, 19, 20, 21, 22, 23, 24]])]
~~~

- `arr.view()` 进行浅拷贝，`arr.copy()` 进行深拷贝

## 数组基础运算

- 同维数组可直接进行四则运算

~~~python
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
data + ones
# array([2, 3])
data - ones
# array([0, 1])
data * data
# array([1, 4])
data / data
# data / data
~~~

- `arr.sum()` 同数组元素求和,也可对高维数组同维元素求和

~~~python
a = np.array([1, 2, 3, 4])
a.sum()
# 10
b = np.array([[1, 1], [2, 2]])
b.sum(axis=0)
# array([3, 3])
b.sum(axis=1)
# array([2, 4])
~~~

## 传播

- 类似数乘

~~~python
data = np.array([1.0, 2.0])
data * 1.6
# array([1.6, 3.2])
~~~

## Reference

- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
