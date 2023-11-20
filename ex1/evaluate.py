# coding=utf-8
import numpy as np


def predict(test_images, theta):
    # test_images[test_images<=40]=0
    # test_images[test_images>40] = 1
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds

def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    y=y.reshape(len(y))
    res = y_pred-y    #相减，相同的项相减后为0
    incorrect = np.count_nonzero(res)   #计算有多少个不为0的项，即y_pred和y不相同的项的个数
    acc = 1-incorrect/len(y)         #计算正确率
    
    return acc