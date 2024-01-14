# coding=utf-8
import numpy as np

def softmax(y_pred,y):
    batch_size ,_ = y_pred.shape
    #y_pred = y_pred / y_pred.max(axis=1)[:,None] #防止溢出
    #y_pred+=1e-5
    y_pred = np.exp(y_pred)
    y_sum = y_pred.sum(axis = 1)
    y_pred = y_pred/y_sum[:,None]
    loss = -np.log(y_pred).T * y
    loss = loss.sum()/batch_size
    grad = y_pred - y.T
    grad /= batch_size
    acc = (y_pred.argmax(axis=1) == y.argmax(axis=0)).mean()
    return loss,grad,acc

def cal_accuracy(model,x_val,y_val):
    y_pred = model.fit(x_val,x_val.shape[0])
    _,_,acc = softmax(y_pred,y_val)
    
    return acc