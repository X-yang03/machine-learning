# coding=utf-8
import numpy as np
import struct
import os

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
    iters = 200
    alpha = 0.75

    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    print("Got data. ") 

    # train the classifier
    theta = train(train_images, train_labels, k, iters, alpha)
    print("Finished training. ") 
    y_predict = predict(test_images, theta)
    accuracy  = cal_accuracy(y_predict, test_labels)
    print("accuracy: {:.2%}".format(accuracy))

    m, n = test_images.shape #(60000,784),28*28 image
    # data processing
    x, y = data_convert(test_images, test_labels, m, k)
    alpha = 0.5
    theta = softmax_regression(theta, x, y, 300, alpha)
    print('second train!')

    # evaluate on the testset
    y_predict = predict(test_images, theta)
    accuracy  = cal_accuracy(y_predict, test_labels)
    print("accuracy: {:.2%}".format(accuracy))
    print("Finished test. ") 
    
