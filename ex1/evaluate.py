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
    res = y_pred-y
    incorrect = np.count_nonzero(res)
    acc = 1-incorrect/len(y)
    
    
    return acc