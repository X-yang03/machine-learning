# coding=utf-8
import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import seaborn as sns

from data_process import load_mnist, load_data
from data_process import data_convert
from LeNet import LeNet
from evaluate import softmax,cal_accuracy
from AdaGrad import AdaGrad
from tqdm import tqdm
    

if __name__ == '__main__':
    # initialize the parameters needed
    mnist_dir = "./mnist_data/"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    

    # get the data
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    print("Got data. ") 

    # train the classifier
    train_images = train_images.astype(float)
    test_images = test_images.astype(float)
    x,y = data_convert(train_images, train_labels,60000,10)
    x_val , y_val = data_convert(test_images,test_labels,10000,10)
    def shuffle_batch(batch_size):

        index = np.random.randint(0,len(x),batch_size)
        return x[index],y.T[index].T
    
    batch_size = 256

    model = LeNet()
    for e in range(10):
        bar = tqdm(range(0, int(x.shape[0]/batch_size)), ncols=100)
        for i in bar:
            #get sample
            X_train,y_train = shuffle_batch(batch_size)
            #forwarding
            y_pred = model.fit(X_train,batch_size)
            #softmax, calculate grad and loss and accuracy
            loss, grad, acc = softmax(y_pred, y_train)
            #backward
            model.back_prop(grad)
            #update the parameters using Adam
            model.update()
            bar.set_postfix(loss=loss, acc=acc)

    # evaluate on the testset
    accuracy  = cal_accuracy(model,x_val, y_val)
    print("accuracy: {:.2%}".format(accuracy))
    print("Finished test. ") 