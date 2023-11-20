# coding=utf-8
import numpy as np
from evaluate import predict, cal_accuracy

from tqdm import tqdm


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    x = x.T  #如同上述，需要将X进行转置为形状()，方便后续运算
    Loss = [] #用于记录每次迭代后的损失值
    for i in tqdm(range(iters)):  #tqdm用于显示迭代进度与速度
        result = theta@x
        res_exp = np.exp(result)
        exp_sum = res_exp.sum(axis=0) #求每列之和，即每个训练样本，在0-9上的计算结果之和

        res_exp = res_exp/exp_sum[None,:] # exp/exp.sum(),每个z_ij除以每列之和，进行softmax
        #res_exp即是y_hat

        loss = np.log(res_exp) * y  #由于res_exp和y形状相同，可以通过二者的对位乘法，计算损失值
        f = -loss.sum()/x.shape[1]  #求取60000个样本的平均损失值，作为模型的损失值
        Loss.append(f)  			#加入Loss列表

        grad = x@(res_exp-y).T 		#如同原理部分所述，计算出梯度矩阵

        if i % 50 == 0 and i>0:		#学习率衰退，每50轮衰减0.98
            alpha = alpha *0.95

        theta = theta - alpha*1/x.shape[1]*grad.T  #更新theta矩阵
    
    return Loss,theta  #此处做了修改，为了方便分析损失值变化情况，会将Loss列表一同返回



