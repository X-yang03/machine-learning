# coding=utf-8
import numpy as np
from evaluate import predict, cal_accuracy

from tqdm import tqdm


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    x = x.T
    y_pos = np.argmax(y,axis=0) #每一个y值（标签）所在的位置(行数)``
    pos_for_loss =[(y_pos[x],x) for x in range(len(y_pos))]
    
    for i in tqdm(range(iters)):
        result = theta@x
        res_exp = np.exp(result)
        exp_sum = res_exp.sum(axis=0) #求每列之和，即每个训练样本，在0-9上的计算结果之和

        # div_by_sum = [exp_sum for i in range(len(res_exp))]
        # div_by_sum = np.array(div_by_sum)
        # res_exp /= div_by_sum   # exp/exp.sum(),softmax

        res_exp = res_exp/exp_sum[None,:]

        ## 接下来计算softmax的损失函数，-log(y)
        #loss = [-np.log(res_exp[x]) for x in pos_for_loss ] #得到每个样例的softmax损失值

        #f = sum(loss)/len(loss) #损失值
        #print(cost)

        grad = x@(res_exp-y).T #梯度

        theta = theta - alpha*1/x.shape[1]*grad.T

        res = res_exp.argmax(axis = 0)
    
    return theta



# def softmax_regression(theta, x, y, iters, alpha, x_t, y_t):
#     # TODO: Do the softmax regression by computing the gradient and
#     # the objective function value of every iteration and update the theta

#     m, n = np.shape(x)
#     theta_T = theta.T
#     y_T = y.T

#     for i in tqdm(range(0, iters)):
#         x_exp = np.exp(x @ theta_T)
#         row_sums = np.sum(x_exp, axis=1)
#         result = x_exp / row_sums[:, None]
        
#         loss  = -np.sum(np.log(result)*y_T)
#         f = loss/m

#         grad = (x.T)@(result-y.T)
#         theta_T = theta_T - alpha*1/m*(grad)
#         theta = theta_T.T

#         if (i+1)%50==0:
#             # evaluate on the testset
#             y_predict = predict(x_t, theta)
#             accuracy = cal_accuracy(y_predict, y_t)
#             # print("Finished test. ")
#             tqdm.write(" iters:%d, accuracy:%2f" % (i+1, accuracy*100))

#     return theta
    
