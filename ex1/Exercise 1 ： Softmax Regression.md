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

本次的手写数字识别任务是一个典型的多分类任务，在这样的情况下某一次预测的交叉熵损失函数计算公式为：
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





## 优化方法

### 提前转换参数为float型大幅提升训练速度



### 学习率$\alpha$衰变



### 正则项的设置
