# coding=utf-8
import numpy as np
from evaluate import predict, cal_accuracy

from tqdm import tqdm


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    x = x.T
    Loss = []
    for i in tqdm(range(iters)):
        result = theta@x
        res_exp = np.exp(result)
        exp_sum = res_exp.sum(axis=0) #求每列之和，即每个训练样本，在0-9上的计算结果之和

        res_exp = res_exp/exp_sum[None,:] # exp/exp.sum(),softmax

        loss = np.log(res_exp) * y
        f = -loss.sum()/x.shape[1]
        Loss.append(f)

        grad = x@(res_exp-y).T #梯度

        # if i%100 == 0 and i>0:
        #     alpha = alpha *0.95

        theta = theta - alpha*1/x.shape[1]*grad.T

        res = res_exp.argmax(axis = 0)
    
    return Loss,theta



