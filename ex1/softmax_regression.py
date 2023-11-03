# coding=utf-8
import numpy as np
from evaluate import predict, cal_accuracy

from tqdm import tqdm


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    x = x.T
    for i in tqdm(range(iters)):
        result = theta@x
        res_exp = np.exp(result)
        exp_sum = res_exp.sum(axis=0) #求每列之和，即每个训练样本，在0-9上的计算结果之和

        div_by_sum = [exp_sum for i in range(len(res_exp))]
        div_by_sum = np.array(div_by_sum)
        res_exp /= div_by_sum   # exp/exp.sum(),softmax

        ## 接下来计算softmax的损失函数，-log(y)
        y_pos = np.argmax(y,axis=0) #每一个y值（标签）所在的位置(行数)``
        pos_for_loss =[(y_pos[x],x) for x in range(len(y_pos))]
        loss = [-np.log(res_exp[x]) for x in pos_for_loss ] #得到每个样例的softmax损失值

        f = sum(loss)/len(loss) #损失值
        #print(cost)

        grad = x@(res_exp-y).T #梯度

        theta = theta - alpha*1/x.shape[1]*grad.T

        res = res_exp.argmax(axis = 0)
        
        # while i%50==0:
        #     alpha -= 0.03
        
        # while(i%50==0):
        #     test_lable=[[y_pos[i]] for i in range(len(y_pos))]
        #     test_lable = np.array(test_lable)
        #     acc = cal_accuracy(res,test_lable)
        #     print("accuracy: {:.2%}".format(acc))
        #     alpha -= 0.03


        # max_row = np.argmax(result, axis=0)  #找到每列最大值的位置（行数）,即是判断结果
        # one_hot_res = np.zeros(result.shape)
        # for col in range(result.shape[1]):   #将每列最大值的位置置1，代表判别结果,one_hot_res与y对应
        #     one_hot_res[max_row[col],col] = 1
        

        # new_theta = np.zeros(theta.T.shape)
    
    return theta
    
