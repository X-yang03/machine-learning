import numpy as np
from layers import *
        
class LeNet:
    def __init__(self):
        self.Conv1 = Conv(1,6,5)    # 卷积层1，输入通道为1，6个5*5的卷积核,输出N*6*24*24
        self.Relu1 = Relu() 
        self.Pool1 = MaxPool(2)     # 池化层1，2*2大小的最大池化,输出N*6*12*12
        self.Conv2 = Conv(6,16,5)   # 卷积层2，6输入通道，16个5*5的卷积核,输出N*16*8*8
        self.Relu2 = Relu()
        self.Pool2 = MaxPool(2)     # 池化层2，2*2大小的最大池化，输出N*16*4*4
        self.Fc1 = Linear(16*4*4,120)   # 全连接层1，16*4*4的输入，120个神经元
        self.Relu3 = Relu()
        self.Fc2 = Linear(120,84)       # 全连接层2，120输入，84个神经元
        self.Relu4 = Relu()
        self.Output = Linear(84,10)     # 输出层，84输入，10个输出

    def fit(self,x,batch_size):
        x = x.reshape((batch_size,1,28,28))
        x = self.Pool1.forward(self.Relu1.forward(self.Conv1.forward(x)))
        x = self.Pool2.forward(self.Relu2.forward(self.Conv2.forward(x)))
        x = x.reshape(batch_size,-1)
        x = self.Relu3.forward(self.Fc1.forward(x))
        x = self.Relu4.forward(self.Fc2.forward(x))
        x = self.Output.forward(x)
        return x
    
    def back_prop(self,grad):
        grad = self.Output.backprop(grad)
        grad = self.Relu4.backprop(grad)
        grad = self.Fc2.backprop(grad)
        grad = self.Relu3.backprop(grad)
        grad = self.Fc1.backprop(grad)
        grad = grad.reshape(grad.shape[0],16,4,4)
        grad = self.Pool2.backprop(grad)
        grad = self.Relu2.backprop(grad)
        grad = self.Conv2.backprop(grad)
        grad = self.Pool1.backprop(grad)
        grad = self.Relu1.backprop(grad)
        grad = self.Conv1.backprop(grad)

    def update(self,alpha):
        self.Output.update_params(alpha)
        self.Fc2.update_params(alpha)
        self.Fc1.update_params(alpha)
        self.Conv2.update_params(alpha)
        self.Conv1.update_params(alpha)
    
    def get_params(self):
        return [self.Conv1.Weight, self.Conv1.Bias, self.Conv2.Weight, self.Conv2.Bias, self.Fc1.Weight, self.Fc1.Bias, 
                self.Fc2.Weight, self.Fc2.Bias, self.Output.Weight, self.Output.Bias]

    def get_grad(self):
        return [self.Conv1.W_grad, self.Conv1.B_grad, self.Conv2.W_grad, self.Conv2.B_grad, self.Fc1.W_grad, self.Fc1.B_grad, 
                self.Fc2.W_grad, self.Fc2.B_grad, self.Output.W_grad, self.Output.B_grad]

    def set_params(self, params):
        self.Conv1.Weight = params[0]
        self.Conv1.Bias = params[1]
        self.Conv2.Weight = params[2]
        self.Conv2.Bias = params[3]
        self.Fc1.Weight = params[4]
        self.Fc1.Bias = params[5]
        self.Fc2.Weight = params[6]
        self.Fc2.Bias = params[7]
        self.Output.Weight = params[8]
        self.Output.Bias = params[9]

