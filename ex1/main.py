# coding=utf-8
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import seaborn as sns

from data_process import load_mnist, load_data
from data_process import data_convert
from train import train
from evaluate import predict, cal_accuracy
from softmax_regression import softmax_regression
    

if __name__ == '__main__':
    # initialize the parameters needed
    mnist_dir = "./mnist_data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    k = 10
    iters = 1000
    alpha = 0.75

    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    print("Got data. ") 

    # train the classifier
    
    l = [1,2,3,4]
    sns.lineplot(x=l,y=l)
    #print((test-test1).sum())
    train_images = train_images /3
    loss,theta = train(train_images, train_labels, k, 100, alpha)
    
    # print("Finished training. ") 
    # y_predict = predict(test_images, theta)
    # accuracy  = cal_accuracy(y_predict, test_labels)
    # print("accuracy: {:.2%}".format(accuracy))

    # m, n = test_images.shape #(60000,784),28*28 image
    # test_images = test_images / 3
    # # data processing
    # alpha = 0.75
    # f,theta = train(test_images, test_labels, k, 1000, alpha)
    #print(f)
    # print('second train!')

    # evaluate on the testset
    y_predict = predict(test_images, theta)
    accuracy  = cal_accuracy(y_predict, test_labels)
    print("accuracy: {:.2%}".format(accuracy))
    print("Finished test. ") 
    
