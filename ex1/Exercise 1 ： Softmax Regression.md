#  Exercise 1 ： Softmax Regression

<center> 朱霞洋 2113301</center>

## 实验原理

### Softmax


在多类别分类问题中，Softmax 函数通常用于神经网络的输出层，将网络的原始输出转化为各个类别的概率分布。在训练过程中，Softmax 函数可以最大化正确类别的概率，并最小化其他类别的概率，从而提高模型的分类性能。它的主要作用是将一组数值转换为概率分布，使得每个数值的范围在 0 到 1 之间，并且它们的总和为 1。

Softmax 函数的定义如下：
$$
\large
p_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
$$
Softmax 函数的输出具有以下特性：

1. 输出的概率在 0 到 1 之间，可视为每个类别的预测概率。
2. 所有类别的概率总和为 1，因此可以解释为一个概率分布。

在多类别分类问题中，Softmax 函数通常用于神经网络的输出层，将网络的原始输出转化为各个类别的概率分布。在训练过程中，Softmax 函数帮助最大化正确类别的概率，并最小化其他类别的概率，从而提高模型的分类性能。



### Softmax 中交叉熵损失函数

在 Softmax 中，可以用交叉熵损失函数来衡量模型输出的概率分布与实际标签之间的差异。交叉熵损失函数是多分类任务中常用的损失函数，尤其在神经网络训练中广泛使用；与 Softmax 结合起来，可以更好地引导模型学习正确的分类概率分布

#### 1. 交叉熵损失的计算方法：

本次的手写数字识别任务是一个典型的多分类任务，在这样的情况下某一次识别的交叉熵损失函数计算公式为：
$$
L(y, \hat{y}) = - \sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)
$$

其中，C 是类别数，$y_i$  是实际标签的独热编码，$\hat{y}_i$ 是模型对类别 $i$ 的预测概率。

又由于代码中训练时的实际标签会被设置为只有正确项$train\_label[i]$为1，其余项为0的独热编码，因此对于某一次预测的交叉熵损失值可以简单由
$$
L(y, \hat{y}) = -  \log(\hat{y}_{train\_lable[i]})
$$
计算得出，其中$train\_label[i]$是这一次预测的真实值；



在本次实验中，训练时共有60000个样本，因此可以由这些样本的交叉熵损失值取平均来代表整个$\theta$矩阵预测的结果和真实结果之间的差别大小。

#### 2. 交叉熵损失的意义：

- **对数似然性质：** 交叉熵损失刻画了实际标签和模型预测之间的负对数似然，最小化交叉熵损失值，等价于最大化模型对实际标签的似然，也就是说最大化模型的准确率；

- **概率分布匹配：** 交叉熵损失的形式促使模型的输出概率分布与实际标签的分布尽可能接近，使得模型能更加专注于正确分类的类别，从而提升准确性。

#### 3. Softmax 结合交叉熵损失：

在 Softmax 中，其将模型的原始输出转化为类别的概率分布，然后交叉熵损失用于比较模型的预测概率与实际标签的概率分布。通过最小化交叉熵损失，模型能够更好地拟合训练数据，提高在手写数字识别任务中的性能。



### 梯度下降

梯度下降是一种迭代优化算法，用于最小化或最大化目标函数。在机器学习和深度学习中，梯度下降主要用于调整模型参数，使得模型能够更好地拟合训练数据或最小化损失函数。

在Lab1中，训练集的输入train_images为矩阵X：
$$
X=\left[
\begin{matrix}
x_{10} & x_{11} & x_{12} & \cdots & x_{1n} \\
x_{20} & x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{m0} & x_{m1} & x_{m2} & \cdots & x_{mn} \\
\end{matrix}
\right]
$$
其中$m = 60000 , n = 784$，这是初始的x的形状，后续训练时会需要转置；

而训练集的标签train_label在经过data_convert()函数进行独热编码后成为矩阵Y：
$$
Y=\left[
\begin{matrix}
y_{11} & y_{12} & \cdots & y_{1m} \\
y_{21} & y_{22} & \cdots & y_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
y_{k1} & y_{k2} & \cdots & y_{km} \\
\end{matrix}
\right]
$$
其中$k=10$,该矩阵的形状为$(10,60000)$，Y的每一列只有一个值为1，其它全为0。$y_{ij}=1 (1\leq i\leq m,1\leq j\leq k)$表示第j个样本对应的类别为第i类。

我们需要优化的参数矩阵为：
$$
\theta = \left[
\begin{matrix}
\omega_{01} & \omega_{02} & \cdots & \omega_{0n} \\
\omega_{11} & \omega_{12} & \cdots & \omega_{1n} \\
\omega_{21} & \omega_{22} & \cdots & \omega_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\omega_{k1} & \omega_{k2} & \cdots & \omega_{kn} \\
\end{matrix}
\right]
$$
其形状为$(10,784)$,

记softmax函数的自变量为：
$$
Z = \left[
\begin{matrix}
z_{11} & z_{12} & \cdots & z_{1k} \\
z_{21} & z_{22} & \cdots & z_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
z_{m1} & z_{m2} & \cdots & z_{mk} \\
\end{matrix}
\right]
$$
Z是模型输入数据的加权求和，即$z_{ij} = \sum_{k=0}^{n}x_{ik}\omega_{kj}$,$Z = X\Omega$。
记
$$
\$$
\hat{Y} = softmax(Z) =
\left[
\begin{matrix}
\hat{y}_{11} & \hat{y}_{12} & \cdots & \hat{y}_{1k} \\
\hat{y}_{21} & \hat{y}_{22} & \cdots & \hat{y}_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{y}_{m1} & \hat{y}_{m2} & \cdots & \hat{y}_{mk} \\
\end{matrix}
\right]
\$$
其中$\hat{y}_{ij} = \frac{e^{z_{ij}}}{\sum_{p=1}^{k}e^{z_{ip}}}$，表示模型眼中第i个样本属于第j类的概率。
定义模型的总代价函数为
\$$
COST(X) = \sum_{i=0}^{m}(-\sum_{j=1}^{k}y_{ij}\ln\hat{y}_{ij})
$$
$$
将代价函数视为参数$\Omega$的函数$J_{\Omega}$,这就是我们要优化的目标。

#### 使用梯度下降求解目标函数极小值

由链式法则可知，


$$

$$

$$
\frac{\partial J_\Omega}{\partial \omega_{pj}} = \sum_{i=1}^{m} \frac{\partial J_\Omega}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial \omega_{pj}}
$$



而
$$
\begin{equation}
\frac{\partial J_\Omega}{\partial z_{ip}}=\frac{\partial \sum_{q=0}^{m}(-\sum_{j=1}^{k}\ln\hat{y}_{qj})}{\partial z_{ip}}= -\sum_{j=1}^{k}\frac{{y}_{ij}}{\hat{y}_{ij}}\frac{\partial\hat{y}_{ij}}{\partial z_{ip}}
\end{equation}
$$
由于
$$
\begin{equation}
\frac{\partial\hat{y}_{ij}}{\partial z_{ip}}=\left\{
\begin{aligned}
& =\frac{e^{z_{ij}}}{\sum_{q=1}^{k}e^{z_{iq}}}-\frac{e^{z_{ij}}e^{z_{ip}}}{(\sum_{q=1}^{k}e^{z_{iq}})^2}=\hat{y}_{ip}-\hat{y}_{ip}^2 &j=p \\
& =-\frac{e^{z_{ij}}e^{z_{ip}}}{(\sum_{q=1}^{k}e^{z_{iq}})^2}=-\hat{y}_{ij}\hat{y_ip} &j\neq p \\
\end{aligned}
\right.
\end{equation}
$$
将(3)式带入(2)式，由于$\sum_{j=1}^{k}y_{ij}=1$,有
$$
\begin{equation}
\frac{\partial J_\Omega}{\partial z_{ip}}=\hat{y}_{ip}-y_{ip}
\end{equation}
$$
考虑$z_{ip}=\sum_{q=0}^{n}x_{iq}\omega_{qp}$，并将(4)式带入(1)式，得到：
$$
\frac{\partial J_\Omega}{\partial \omega_{pj}}=\sum_{i=1}^{m}\frac{\partial J_\Omega}{\partial z_{ij}}x_{ip}=\sum_{i=1}^{m}(\hat{y}_{ij}-y_{ij})x_{ip}
$$
写回矩阵的形式，有：
$$
\nabla J_\Omega = \left[
\begin{matrix}
\frac{J_\Omega}{\omega_{01}} &\frac{J_\Omega}{\omega_{02}} & \cdots & \frac{J_\Omega}{\omega_{0k}} \\
\frac{J_\Omega}{\omega_{11}} &\frac{J_\Omega}{\omega_{12}} & \cdots & \frac{J_\Omega}{\omega_{1k}} \\
\vdots & \vdots & \ddots &\vdots \\
\frac{J_\Omega}{\omega_{n1}} &\frac{J_\Omega}{\omega_{n2}} & \cdots & \frac{J_\Omega}{\omega_{nk}} \\
\end{matrix}
\right]=\left[
\begin{matrix}
\sum_{i=1}^{m}x_{i0}(\hat{y}_{i1}-y_{i1}) & \sum_{i=1}^{m}x_{i0}(\hat{y}_{i2}-y_{i2}) & \cdots & \sum_{i=1}^{m}x_{i0}(\hat{y}_{ik}-y_{ik})\\
\sum_{i=1}^{m}x_{i1}(\hat{y}_{i1}-y_{i1}) & \sum_{i=1}^{m}x_{i1}(\hat{y}_{i2}-y_{i2}) & \cdots & \sum_{i=1}^{m}x_{i1}(\hat{y}_{ik}-y_{ik})\\
\vdots & \vdots & \ddots & \cdots \\
\sum_{i=1}^{m}x_{in}(\hat{y}_{i1}-y_{i1}) & \sum_{i=1}^{m}x_{in}(\hat{y}_{i2}-y_{i2}) & \cdots & \sum_{i=1}^{m}x_{in}(\hat{y}_{ik}-y_{ik})\\
\end{matrix}
\right]
$$
梯度已经求出来了，指定步长使用梯度下降方法求解即可。



梯度下降通过迭代更新模型的参数，使得目标函数逐渐趋近于最小值或最大值。参数更新的一般规则为：
$$
\nabla J(\theta) = \left[ \frac{\partial J(\theta)}{\partial \theta_1}, \frac{\partial J(\theta)}{\partial \theta_2}, \ldots, \frac{\partial J(\theta)}{\partial \theta_n} \right]
$$
其中，�*α* 是学习率，∇�(��)∇*J*(*θ**t*) 是目标函数在当前参数位置的梯度。
$$
 \theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t) 
$$


### 5. 批量梯度下降和小批量梯度下降：

- **批量梯度下降（Batch Gradient Descent）：** 在每次迭代中，使用整个训练数据计算梯度。
- **小批量梯度下降（Mini-Batch Gradient Descent）：** 每次迭代使用部分训练数据计算梯度，通常是随机选择的一小部分样本。
- **随机梯度下降（Stochastic Gradient Descent）：** 每次迭代使用一个随机样本计算梯度。

### 6. 收敛条件：

梯度下降迭代更新参数，直到满足某个停止条件，例如达到最大迭代次数、目标函数的变化小于某个阈值等。

梯度下降是优化问题中的基础算法，广泛应用于各种机器学习和深度学习模型的训练过程中。



## 程序设计

本次实验主要完成softmax_regression和 cal_accuracy两个函数的设计：

#### softmax_regression

```python
def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    
    x = x.T   #x的形状原本为(60000,784),转置后为(784,60000)，可以与theta进行点乘
    Loss = [] #存储每轮迭代时的交叉熵损失值
    for i in tqdm(range(iters)):  #tqdm用于展现迭代进度
        result = theta@x   #theta的形状为(10,784)，点乘x后result形状为(10,60000)
        
        res_exp = np.exp(result)	#进行求指数操作
        
        exp_sum = res_exp.sum(axis=0) #求每列之和，即每个训练样本，在0-9上的计算结果之和

        res_exp = res_exp/exp_sum[None,:] # exp/exp.sum(),每个值除以其所在列之和，进行softmax

        loss = np.log(res_exp) * y #由于res_exp和y的形状都为(10,60000),可以由矩阵对位乘法获得其交叉熵损失值
        f = -loss.sum()/x.shape[1] #将60000个样本的交叉熵损失值进行求均值
        Loss.append(f)				#加入到Loss列表中

        grad = x@(res_exp-y).T #计算梯度

        if i % 50 == 0 and i>0:		#学习率衰败，每50轮乘以0.98
            alpha = alpha *0.98

        theta = theta - alpha*1/x.shape[1]*grad.T   #theta的迭代
    
    return Loss,theta #为了方便分析，此处做了修改，将Loss列表一同返回
```



#### cal_accuracy

```python
def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    y=y.reshape(len(y))
    res = y_pred-y  #y_pred和y相减，相同的项为0
    incorrect = np.count_nonzero(res)# 计算不为0的项的个数，即是识别错误项的个数
    acc = 1-incorrect/len(y)  #正确率
    
    return acc
```



$$
X=\left[
\begin{matrix}
x_{10} & x_{11} & x_{12} & \cdots & x_{1n} \\
x_{20} & x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{m0} & x_{m1} & x_{m2} & \cdots & x_{mn} \\
\end{matrix}
\right]
$$


$\mathbf{X}$



$\mathbf{X}_{ij}(1\leq i\leq m,1\leq j\leq n)$


$$
Y=\left[
\begin{matrix}
y_{11} & y_{12} & \cdots & y_{1k} \\
y_{21} & y_{22} & \cdots & y_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
y_{m1} & y_{m2} & \cdots & y_{mk} \\
\end{matrix}
\right]
$$
Y的每一行只有一个值为1，其它全为0。$y_{ij}=1 (1\leq i\leq m,1\leq j\leq k)$表示第i个样本对应的类别为第j类。
记待估计的参数为
$$
\Omega = \left[
\begin{matrix}
\omega_{01} & \omega_{02} & \cdots & \omega_{0k} \\
\omega_{11} & \omega_{12} & \cdots & \omega_{1k} \\
\omega_{21} & \omega_{22} & \cdots & \omega_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
\omega_{n1} & \omega_{n2} & \cdots & \omega_{nk} \\
\end{matrix}
\right]
$$
记softmax函数的自变量为：
$$
Z = \left[
\begin{matrix}
z_{11} & z_{12} & \cdots & z_{1k} \\
z_{21} & z_{22} & \cdots & z_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
z_{m1} & z_{m2} & \cdots & z_{mk} \\
\end{matrix}
\right]
$$
Z是模型输入数据的加权求和，即$z_{ij} = \sum_{k=0}^{n}x_{ik}\omega_{kj}$,$Z = X\Omega$。
记
$$
\$$
\hat{Y} = softmax(Z) =
\left[
\begin{matrix}
\hat{y}_{11} & \hat{y}_{12} & \cdots & \hat{y}_{1k} \\
\hat{y}_{21} & \hat{y}_{22} & \cdots & \hat{y}_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{y}_{m1} & \hat{y}_{m2} & \cdots & \hat{y}_{mk} \\
\end{matrix}
\right]
\$$
其中$\hat{y}_{ij} = \frac{e^{z_{ij}}}{\sum_{p=1}^{k}e^{z_{ip}}}$，表示模型眼中第i个样本属于第j类的概率。
定义模型的总代价函数为
$$
COST(X) = \sum_{i=0}^{m}(-\sum_{j=1}^{k}y_{ij}\ln\hat{y}_{ij})
$$
$$
将代价函数视为参数$\Omega$的函数$J_{\Omega}$,这就是我们要优化的目标。

#### 使用梯度下降求解目标函数极小值

由链式法则可知，


$$

$$

$$
\frac{\partial J_\Omega}{\partial \omega_{pj}} = \sum_{i=1}^{m} \frac{\partial J_\Omega}{\partial z_{ij}}\frac{\partial z_{ij}}{\partial \omega_{pj}}
$$


$$

$$


```

```

而
$$
\begin{equation}
\frac{\partial J_\Omega}{\partial z_{ip}}=\frac{\partial \sum_{q=0}^{m}(-\sum_{j=1}^{k}\ln\hat{y}_{qj})}{\partial z_{ip}}= -\sum_{j=1}^{k}\frac{{y}_{ij}}{\hat{y}_{ij}}\frac{\partial\hat{y}_{ij}}{\partial z_{ip}}
\end{equation}
$$
由于
$$
\begin{equation}
\frac{\partial\hat{y}_{ij}}{\partial z_{ip}}=\left\{
\begin{aligned}
& =\frac{e^{z_{ij}}}{\sum_{q=1}^{k}e^{z_{iq}}}-\frac{e^{z_{ij}}e^{z_{ip}}}{(\sum_{q=1}^{k}e^{z_{iq}})^2}=\hat{y}_{ip}-\hat{y}_{ip}^2 &j=p \\
& =-\frac{e^{z_{ij}}e^{z_{ip}}}{(\sum_{q=1}^{k}e^{z_{iq}})^2}=-\hat{y}_{ij}\hat{y_ip} &j\neq p \\
\end{aligned}
\right.
\end{equation}
$$
将(3)式带入(2)式，由于$\sum_{j=1}^{k}y_{ij}=1$,有
$$
\begin{equation}
\frac{\partial J_\Omega}{\partial z_{ip}}=\hat{y}_{ip}-y_{ip}
\end{equation}
$$
考虑$z_{ip}=\sum_{q=0}^{n}x_{iq}\omega_{qp}$，并将(4)式带入(1)式，得到：
$$
\frac{\partial J_\Omega}{\partial \omega_{pj}}=\sum_{i=1}^{m}\frac{\partial J_\Omega}{\partial z_{ij}}x_{ip}=\sum_{i=1}^{m}(\hat{y}_{ij}-y_{ij})x_{ip}
$$
写回矩阵的形式，有：
$$
\nabla J_\Omega = \left[
\begin{matrix}
\frac{J_\Omega}{\omega_{01}} &\frac{J_\Omega}{\omega_{02}} & \cdots & \frac{J_\Omega}{\omega_{0k}} \\
\frac{J_\Omega}{\omega_{11}} &\frac{J_\Omega}{\omega_{12}} & \cdots & \frac{J_\Omega}{\omega_{1k}} \\
\vdots & \vdots & \ddots &\vdots \\
\frac{J_\Omega}{\omega_{n1}} &\frac{J_\Omega}{\omega_{n2}} & \cdots & \frac{J_\Omega}{\omega_{nk}} \\
\end{matrix}
\right]=\left[
\begin{matrix}
\sum_{i=1}^{m}x_{i0}(\hat{y}_{i1}-y_{i1}) & \sum_{i=1}^{m}x_{i0}(\hat{y}_{i2}-y_{i2}) & \cdots & \sum_{i=1}^{m}x_{i0}(\hat{y}_{ik}-y_{ik})\\
\sum_{i=1}^{m}x_{i1}(\hat{y}_{i1}-y_{i1}) & \sum_{i=1}^{m}x_{i1}(\hat{y}_{i2}-y_{i2}) & \cdots & \sum_{i=1}^{m}x_{i1}(\hat{y}_{ik}-y_{ik})\\
\vdots & \vdots & \ddots & \cdots \\
\sum_{i=1}^{m}x_{in}(\hat{y}_{i1}-y_{i1}) & \sum_{i=1}^{m}x_{in}(\hat{y}_{i2}-y_{i2}) & \cdots & \sum_{i=1}^{m}x_{in}(\hat{y}_{ik}-y_{ik})\\
\end{matrix}
\right]
$$
梯度已经求出来了，指定步长使用梯度下降方法求解即可。



## 优化方法

### 提前转换参数为float型大幅提升训练速度



### 学习率$\alpha$衰变



### 正则项的设置
