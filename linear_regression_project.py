
# coding: utf-8

# In[29]:


# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 9999


# # 欢迎来到线性回归项目
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励你带着问题提交项目，评审人会给予你诸多帮助。
# 
# 所有选做题都可以不做，不影响项目通过。如果你做了，那么项目评审会帮你批改，也会因为选做部分做错而判定为不通过。
# 
# 其中非代码题可以提交手写后扫描的 pdf 文件，或使用 Latex 在文档中直接回答。

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[5]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何NumPy以及相关的科学计算库来完成作业


# 本项目要求矩阵统一使用二维列表表示，如下：
A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
def get_MxN_Matrix(m,n,x):
    return [[x for i in range(m)] for j in range(n)]

dimension = 4
matrix = get_MxN_Matrix(4,4,0)
for i in xrange(dimension):
    matrix[i][i] = 1
print matrix


# ## 1.2 返回矩阵的行数和列数

# In[6]:


# TODO 返回矩阵的行数和列数
def shape(M):
    rows = len(M)
    cols = len(M[0])
    return rows, cols


# In[7]:


# 运行以下代码测试你的 shape 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[8]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    rowLen = len(M)
    for i in xrange(rowLen):
        colLen = len(M[i])
        for j in xrange(colLen):
            rounded_element = round(M[i][j], decPts)
            M[i][j] = rounded_element
    


# In[9]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[3]:


# TODO 计算矩阵的转置
def get_MxN_Matrix(m,n,x):
    return [[x for i in range(m)] for j in range(n)]

def transpose(M):
    rows = len(M)
    cols = len(M[0])
    t_M = get_MxN_Matrix(rows, cols, 0)
    for row in xrange(cols):
        for col in xrange(rows):
             t_M[row][col] = M[col][row]
                
    return t_M


# In[4]:


# 运行以下代码测试你的 transpose 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[5]:


# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def get_MxN_Matrix(m,n,x):
    return [[x for i in range(m)] for j in range(n)]

def matxMultiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0]) 
    
    if (cols_A != rows_B):
        raise ValueError
    
    # Solution 1
#     AB = get_MxN_Matrix(cols_B, rows_A, 0)
#     for row in xrange(rows_A):
#         for col in xrange(cols_B):
#             for index in xrange(cols_A):
#                 AB[row][col] += A[row][index]*B[index][col]
             
    # Solution 2
    AB = [[(sum(a*b for a,b in zip(tuple_A, tuple_B))) 
           for tuple_B in zip(*B)] 
              for tuple_A in A]
    
    return AB


# In[6]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[7]:


# TODO 构造增广矩阵，假设A，b行数相同
def get_MxN_Matrix(m,n,x):
    return [[x for i in range(m)] for j in range(n)]

import copy

def augmentMatrix(A, b):
    # Solution 1: Running Time = 0.013s
    rows = len(A)
    cols = len(A[0]) + len(b[0])
    Ab = get_MxN_Matrix(cols, rows, 0)
    for i in xrange(rows):
        for j in xrange(cols):
            if j==(cols-1):
                Ab[i][j] = b[i][0]
            else:
                Ab[i][j] = A[i][j]
    
    # Solution 2: Running Time = 0.026s
#     Ab = copy.deepcopy(A)
#     for i in xrange(len(Ab)):
#         Ab[i].append(b[i][0])
    
    return Ab


# In[8]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[9]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    l = M[r1]
    M[r1] = M[r2]
    M[r2] = l


# In[10]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[11]:


# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
        
    for i in xrange(len(M[r])):
        M[r][i] *= scale


# In[12]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[13]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    if scale == 0:
        raise ValueError
        
    for i in xrange(len(M[r1])):
        M[r1][i] += M[r2][i]*scale


# In[14]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 2.3.1 算法
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None (你可以在选做问题2.4中证明为什么这里A一定是奇异矩阵)
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# **注：** 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# ### 2.3.2 算法推演
# 
# ## 备注：手动计算推演太复杂，详细步骤省略。
# 
# 为了充分了解Gaussian Jordan消元法的计算流程，请根据Gaussian Jordan消元法，分别手动推演矩阵A为***可逆矩阵***，矩阵A为***奇异矩阵***两种情况。

# In[15]:


# 不要修改这里！
from helper import *

# A = generateMatrix(4,seed,singular=False)
A = generateMatrix(4,4,singular=False)
b = np.ones(shape=(4,1)) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # please make sure you already correct implement augmentMatrix
printInMatrixFormat(Ab,padding=4,truncating=0)


# 请按照算法的步骤3，逐步推演***可逆矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵。
# 
# 要求：
# 1. 做分数运算
# 2. 使用`\frac{n}{m}`来渲染分数，如下：
#  - $\frac{n}{m}$
#  - $-\frac{a}{b}$
# 
# 增广矩阵
# $ Ab = \begin{bmatrix}
#     4  & -5 & -9 & -2 & 1\\
#     -2 & 8  & -1 & -3 & 1\\
#     3  & -2 & -6 & 8  & 1\\
#     2  & -4 & 0  & -7 & 1\end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\end{bmatrix}$
#     
# $...$

# In[24]:


# 不要修改这里！
# A = generateMatrix(4,seed,singular=True)
A = generateMatrix(4,4,singular=True)
b = np.ones(shape=(4,1)) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # please make sure you already correct implement augmentMatrix
printInMatrixFormat(Ab,padding=4,truncating=0)


# 请按照算法的步骤3，逐步推演***奇异矩阵***的变换。
# 
# 在下面列出每一次循环体执行之后的增广矩阵。
# 
# 要求：
# 1. 做分数运算
# 2. 使用`\frac{n}{m}`来渲染分数，如下：
#  - $\frac{n}{m}$
#  - $-\frac{a}{b}$
# 
# 增广矩阵
# $ Ab = \begin{bmatrix}
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\end{bmatrix}$
# 
# $ --> \begin{bmatrix}
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\end{bmatrix}$
#     
# $ --> \begin{bmatrix}
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\\
#     0 & 0 & 0 & 0 & 0\end{bmatrix}$
#     
# $...$

# ### 2.3.3 实现 Gaussian Jordan 消元法

# In[11]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

from decimal import Decimal, getcontext

getcontext().prec = 30

# row operation 1
def swapRows(M, r1, r2):
    if r1 == r2:
        return
    
    l = M[r1]
    M[r1] = M[r2]
    M[r2] = l

    
# row operation 2
def scaleRow(M, r, scale):
    if is_near_zero(scale):
        raise ValueError
        
    for i in xrange(len(M[r])):
        M[r][i] = Decimal(scale * M[r][i])


# row operation 3
def addScaledRow(M, r1, r2, scale):
    if is_near_zero(scale):
        raise ValueError
        
    for i in xrange(len(M[r1])):
        M[r1][i] += Decimal(M[r2][i]*scale)


def augmentMatrix(A, b):
    # Solution 1: Running Time = 0.013s
    rows = len(A)
    cols = len(A[0]) + len(b[0])
    Ab = get_RxC_Matrix(rows, cols, 0)
    for i in xrange(rows):
        for j in xrange(cols):
            if j==(cols-1):
                Ab[i][j] = b[i][0]
            else:
                Ab[i][j] = A[i][j]
    return Ab


def maxAbsIndex(matrix, col):
    rows = len(matrix)
    cols = len(matrix[0])
    
    if col >= cols:
        return None
    
    maxAbs = 0
    index = col
    for i in xrange(col, rows):
        currentAbs = abs(matrix[i][col])
        if currentAbs > maxAbs:
            maxAbs = currentAbs
            index = i
            
    return maxAbs, index

def get_RxC_Matrix(rows,cols,x):
    return [[x for i in range(cols)] for j in range(rows)]

def getLastCol(Ab, decPts=4):
    rows = len(Ab)
    lastCol = get_RxC_Matrix(rows,1,0)
    for i in xrange(rows):
        lastCol[i][0] = round(Ab[i][len(Ab[0])-1], decPts)
    return lastCol

def is_near_zero(value, epsilon = 1.0e-16):
    return abs(value) < epsilon


def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    # Step 1: Check if A and b have same number of rows
    if (len(A) != len(b)):
        return None
    
    # Step 2: Construct augmented matrix Ab
    Ab = augmentMatrix(A, b)
    
    # Step 3: Column by column, transform Ab to RREF(reduced row echelon form)
    rows = len(A)
    cols = len(A[0])
    for col in xrange(cols):
        # Find in column c, at diagonal and under diagonal (row c ~ N) the maximum absolute value
        maxAbs, index = maxAbsIndex(Ab, col)
        if is_near_zero(maxAbs, epsilon):
            # If the maximum absolute value is 0, then A is singular, return None
            return None
        else:
            # Apply row operation 1, swap the row of maximum with the row of diagonal element (row c)
            swapRows(Ab, col, index)
            
            # Apply row operation 2, scale the diagonal element of column c to 1
            scaleRow(Ab, col, Decimal(1.0)/Decimal(Ab[col][col]))
            
            # Apply row operation 3 mutiple time, eliminate every other element in column c
            for scaleRowIndex in xrange(rows):
                if scaleRowIndex == col:
                    continue
                
                scale = Decimal(-Ab[scaleRowIndex][col])
                if is_near_zero(scale):
                    continue
                else:
                    addScaledRow(Ab, scaleRowIndex, col, scale)
    x = getLastCol(Ab, decPts)
    return x


# In[34]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


# ## (选做) 2.4 算法正确判断了奇异矩阵：
# 
# 在算法的步骤3 中，如果发现某一列对角线和对角线以下所有元素都为0，那么则断定这个矩阵为奇异矩阵。
# 
# 我们用正式的语言描述这个命题，并证明为真。
# 
# 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 证明：
# 
# Since $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} $  , then 
# $ |A| = \begin{vmatrix}
#     I    & X \\ 
#     Z    & Y \\
# \end{vmatrix} $ 
# = |IY| - |XZ|
# 
# I is the identity matrix $\rightarrow$ |IY| = |Y|.  
# Z is all zero $\rightarrow$ |XZ| = 0.  
# Thus, |A| = |Y|  
# 
# The first column of Y is all zero $\rightarrow
# |Y|= \begin{vmatrix}
#     0    & Y1 \\ 
#     O    & Y2 \\
# \end{vmatrix} = 0\times|Y2|-|O||Y1|=0 
# \rightarrow 
# |A| = 0 
# \rightarrow $ A is singular

# # 3  线性回归

# ## 3.1 随机生成样本点

# In[12]:


# 不要修改这里！
# 运行一次就够了！
from helper import *
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

# X,Y = generatePoints(seed, num=100)
X,Y, actual_m, actual_b = generatePoints(seed=10, num=100)

## 可视化
plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b') # b 是blue的意思；color=blue
plt.show()


# ## 3.2 拟合一条直线
# 
# ### 3.2.1 猜测一条直线

# In[8]:


#TODO 请选择最适合的直线 y = mx + b
m = 2.7
b = 5

# 不要修改这里！
plt.xlim((-5,5))
x_vals = plt.axes().get_xlim()
y_vals = [m*x+b for x in x_vals]
plt.plot(x_vals, y_vals, '-', color='r')

plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')

plt.show()


# ### 3.2.2 计算平均平方误差 (MSE)

# 我们要编程计算所选直线的平均平方误差(MSE), 即数据集中每个点到直线的Y方向距离的平方的平均数，表达式如下：
# $$
# MSE = \frac{1}{n}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$

# In[14]:


# TODO 实现以下函数并输出所选直线的MSE
from decimal import Decimal, getcontext

getcontext().prec = 30

def calculateMSE(X,Y,m,b):
    if len(X) != len(Y):
        return None
    
    n = len(X)
    total = 0
    for i in xrange(n):
        total += ((Y[i] - m*X[i] - b)**2)
        
    mse = Decimal(total)/Decimal(n)
#     mse = round(mse, 4)
    return mse

print(calculateMSE(X,Y,m,b))


# ### 3.2.3 调整参数 $m, b$ 来获得最小的平方平均误差
# 
# 你可以调整3.2.1中的参数 $m,b$ 让蓝点均匀覆盖在红线周围，然后微调 $m, b$ 让MSE最小。

# ## 3.3 (选做) 找到参数 $m, b$ 使得平方平均误差最小
# 
# **这一部分需要简单的微积分知识(  $ (x^2)' = 2x $ )。因为这是一个线性代数项目，所以设为选做。**
# 
# 刚刚我们手动调节参数，尝试找到最小的平方平均误差。下面我们要精确得求解 $m, b$ 使得平方平均误差最小。
# 
# 定义目标函数 $E$ 为
# $$
# E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 因为 $E = \frac{n}{2}MSE$, 所以 $E$ 取到最小值时，$MSE$ 也取到最小值。要找到 $E$ 的最小值，即要找到 $m, b$ 使得 $E$ 相对于 $m$, $E$ 相对于 $b$ 的偏导数等于0. 
# 
# 因此我们要解下面的方程组。
# 
# $$
# \begin{cases}
# \displaystyle
# \frac{\partial E}{\partial m} =0 \\
# \\
# \displaystyle
# \frac{\partial E}{\partial b} =0 \\
# \end{cases}
# $$
# 
# ### 3.3.1 计算目标函数相对于参数的导数
# 首先我们计算两个式子左边的值
# 
# 证明/计算：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = bx
# $$

# TODO 证明:  
# 
# $$
# \begin{multline*}
# 1. 证明  \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# \end{multline*}
# $$
# 
# $$ 
# 假设：
# \begin{multline}
# \begin{cases}
# \displaystyle
# E = E(u) = \frac{1}{2} \sum_{i=1}^{n}u^2
# \\
# \displaystyle
# u = -mx_{i}-(b-y_{i})
# \end{cases}   \\
# \end{multline}
# $$
# 
# $$
# \begin{multline}
# 通过复合函数求导法则：\frac{\partial E}{\partial m}={E_{m}}'={E_{u}}'\cdot{u_{m}}' \\
# \end{multline}
# $$
# 
# $$
# \begin{align*}
# \begin{cases}
# {E_{u}}' &= ( \frac{1}{2} \sum_{i=1}^{n}u^2)'= \sum_{i=1}^{n}u= \sum_{i=1}^{n}(y_{i}-mx_{i}-b)) \\
# {u_{m}}' &=   (-mx_{i}-(b+y_{i}))'=-x_{i}\\
# \end{cases}  \\
# \end{align*}
# $$
# 
# $$
# \rightarrow \frac{\partial E}{\partial m}=  \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{multline*}
# 2. 证明  \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# \end{multline*}
# $$
# 
# $$ 
# 假设：
# \begin{multline}
# \begin{cases}
# \displaystyle
# E = E(u) = \frac{1}{2} \sum_{i=1}^{n}u^2
# \\
# \displaystyle
# u = -mx_{i}-(b-y_{i})
# \end{cases}   \\
# \end{multline}
# $$
# 
# $$
# \begin{multline}
# 通过复合函数求导法则：\frac{\partial E}{\partial b}={E_{b}}'={E_{u}}'\cdot{u_{b}}' \\
# \end{multline}
# $$
# 
# $$
# \begin{align*}
# \begin{cases}
# {E_{u}}' &= ( \frac{1}{2} \sum_{i=1}^{n}u^2)'= \sum_{i=1}^{n}u= \sum_{i=1}^{n}(y_{i}-mx_{i}-b)) \\
# {u_{b}}' &=   (-mx_{i}-(b+y_{i}))'=-x_{i}\\
# \end{cases}  \\
# \end{align*}
# $$
# 
# $$
# \rightarrow \frac{\partial E}{\partial b}=  \sum_{i=1}^{n}{-(y_i - mx_i - b)}
# $$
# 

# ### 3.3.2 实例推演
# 
# 现在我们有了一个二元二次方程组
# 
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# $$
# 
# 为了加强理解，我们用一个实际例子演练。
# 
# 我们要用三个点 $(1,1), (2,2), (3,2)$ 来拟合一条直线 y = m*x + b, 请写出
# 
# - 目标函数 $E$, 
# - 二元二次方程组，
# - 并求解最优参数 $m, b$

# TODO 写出目标函数，方程组和最优参数  
# 
# 1. $E = \frac{1}{2}\sum_{i=1}^{n}{(y_i - mx_i - b)^2}$
# 2. 二元二次方程组
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# \rightarrow
# 带入 (1,1), (2,2), (3,2)  
# \rightarrow
# \begin{cases}
# \displaystyle
# \ (-1-4-6)+(m+4m+9m)+(b+2b+3b)=0 
# \\
# \displaystyle
# \ (-1-2-2)+(m+2m+3m)+(b+b+b)=0
# \end{cases}
# \\
# \rightarrow
# \begin{cases}
# \ 6b+14m=11
# \\
# \ 3b+6m=5
# \end{cases}
# \rightarrow 得到增广矩阵 
# \begin{bmatrix}
# 6 & 14 & 11  \\
# 3 & 6  & 5
# \end{bmatrix}
# \rightarrow 经过高斯消元法得到RREF \rightarrow
# \begin{bmatrix}
# 1 & 0 & \frac{2}{3}  \\
# 0 & 1 & \frac{1}{2}
# \end{bmatrix}
# $$
# 
# 3. 求解最有参数m,b  
# $m=\frac{1}{2}$  
# $b=\frac{2}{3}$

# ### 3.3.3 将方程组写成矩阵形式
# 
# 我们的二元二次方程组可以用更简洁的矩阵形式表达，将方程组写成矩阵形式更有利于我们使用 Gaussian Jordan 消元法求解。
# 
# 请证明 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = X^TXh - X^TY
# $$
# 
# 其中向量 $Y$, 矩阵 $X$ 和 向量 $h$ 分别为 :
# $$
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 证明:
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = X^TXh - X^TY
# $$
# 
# $$
# \begin{align*}
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} 
# & = 
# \begin{bmatrix}
#     \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} \\
#     \sum_{i=1}^{n}{-(y_i - mx_i - b)} 
# \end{bmatrix} 
# \\ & =
# \begin{bmatrix}
#     \sum_{i=1}^{n}{mx_{i}^2} + \sum_{i=1}^{n}{bx_{i}} \\
#     \sum_{i=1}^{n}{mx_{i}} + \sum_{i=1}^{n}b
# \end{bmatrix} 
# -
# \begin{bmatrix}
#      \sum_{i=1}^{n}{x_{i}y_{i}} \\
#      \sum_{i=1}^{n}{y_{i}}
# \end{bmatrix}
# \\ & =
# \begin{bmatrix}
#     \sum_{i=1}^{n}{x_{i}^2} &   \sum_{i=1}^{n}{x_{i}} \\
#     \sum_{i=1}^{n}{x_{i}}   &   \sum_{i=1}^{n}1
# \end{bmatrix} 
# \cdot
# \begin{bmatrix}
#     m \\
#     b
# \end{bmatrix}
# -
# \begin{bmatrix}
#      \sum_{i=1}^{n}{x_{i}y_{i}} \\
#      \sum_{i=1}^{n}{y_{i}}
# \end{bmatrix}
# \\ & =
# \begin{bmatrix}
#     x_{1}^2+x_{2}^2+...x_{n}^2  &   x_{1}+x_{2}+...x_{n} \\
#      x_{1}+x_{2}+...x_{n}       &   n
# \end{bmatrix} 
# \cdot
# \begin{bmatrix}
#     m \\
#     b
# \end{bmatrix}
# -
# \begin{bmatrix}
#      x_{1}y_{1}+x_{2}y_{2}+x_{3}y_{3} ...+x_{n}y_{n} \\
#      y_{1}+y_{2}+y_{3}+...+y_{n}
# \end{bmatrix}
# \\ & =
# \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1   &  1  & ... & 1
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
#     m \\
#     b
# \end{bmatrix}
# -
# \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1   &  1  & ... & 1
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# \\ & =
# \ X^TXh - X^TY
# \end{align*}
# $$

# 至此我们知道，通过求解方程 $X^TXh = X^TY$ 来找到最优参数。这个方程十分重要，他有一个名字叫做 **Normal Equation**，也有直观的几何意义。你可以在 [子空间投影](http://open.163.com/movie/2010/11/J/U/M6V0BQC4M_M6V2AJLJU.html) 和 [投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html) 看到更多关于这个方程的内容。

# ### 3.4 求解 $X^TXh = X^TY$ 
# 
# 在3.3 中，我们知道线性回归问题等价于求解 $X^TXh = X^TY$ (如果你选择不做3.3，就勇敢的相信吧，哈哈)
# 
# #### 求解方法
# 1. 根据公式构建增广矩阵  
# $$
# \begin{cases}
# \displaystyle
# \sum_{i=1}^{n}{-x_i(y_i - mx_i - b)} =0 \\
# \\
# \displaystyle
# \sum_{i=1}^{n}{-(y_i - mx_i - b)} =0 \\
# \end{cases}
# \rightarrow
# \begin{cases}
# \ m(\sum_{i=1}^{n}x_i^2) + b(\sum_{i=1}^{n}x_i) = \sum_{i=1}^{n}(x_iy_i)
# \\
# \ m(\sum_{i=1}^{n}x_i) + b(\sum_{i=1}^{n}1) = \sum_{i=1}^{n}(y_i)
# \end{cases}
# $$
# 
# \begin{align*}
# 列顺序：&m \ + \ b = VALUE  \\
# &\begin{bmatrix}
#     \sum_{i=1}^{n}x_i^2  &  \sum_{i=1}^{n}x_i  &  \sum_{i=1}^{n}(x_iy_i)  \\
#     \sum_{i=1}^{n}x_i    &  \sum_{i=1}^{n}1    &  \sum_{i=1}^{n}(y_i)
# \end{bmatrix}
# \end{align*}
# 
# 对增广矩阵进行Gaussian Jordan可以求得 $m,b$ 的值

# In[21]:


# TODO 实现线性回归
'''
参数：X, Y
返回：m，b
'''
from decimal import Decimal, getcontext

getcontext().prec = 30
    
def sum_of_xi_yi(X,Y):
    n = len(X)
    total = 0
    for i in xrange(n):
        total += Decimal(X[i])*Decimal(Y[i])
    return total

def linearRegression(X,Y):
    xi_squared = sum(Decimal(x)**2 for x in X)
    xi = sum(Decimal(x) for x in X)
    xi_yi = sum_of_xi_yi(X,Y)
    yi = sum(Decimal(y) for y in Y)
    n = len(X)
    
    A = [[xi_squared, xi],[xi, n]]
    b = [[xi_yi], [yi]]
    h = gj_Solve(A, b)
    m = h[0][0]
    b = h[1][0]
    return m, b

m,b = linearRegression(X,Y)
print(m,b)


# 你求得的回归结果是什么？
# 请使用运行以下代码将它画出来。

# In[22]:


# 请不要修改下面的代码
x1,x2 = -5,5
y1,y2 = x1*m+b, x2*m+b

plt.xlim((-5,5))
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.scatter(X,Y,c='b')
plt.plot((x1,x2),(y1,y2),'r')
plt.text(1,2,'y = {m}x + {b}'.format(m=m,b=b))
plt.show()


# 你求得的回归结果对当前数据集的MSE是多少？

# In[23]:


MSE = calculateMSE(X,Y,m,b)
print "MSE={}".format(MSE)

