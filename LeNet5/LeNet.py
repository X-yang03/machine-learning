import numpy as np

class Relu:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backprop(self,grad):
        return np.where(self.x > 0, 1, 0) * grad
    

class Conv:
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
        """
        params: in_channels: the number of channels of the input image
                out_channels: the number of channels of the output image
                filter_size:(x,y) the size of the filter
                stride: the stride of the filter
                padding: the padding of the filter
        """
        self.input = None
        self.output = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.Weight = np.random.normal(scale=1, size=(out_channels,in_channels,filter_size,filter_size))
        #第一个维度是输出通道数，对应着有这么多个卷积核；第二个维度是输入通道数，这是因为卷积核需要负责讲in个channel的输入变成out个channel的输出
        self.W_grad = np.zeros_like(self.Weight)
        self.Bias = np.zeros(out_channels) #每个卷积核有一个偏置参数
        self.B_grad = np.zeros_like(self.Bias)
    
    def conv2d(self,input,kernel,padding,Bias = True):
        N,C,H,W = input.shape
        if padding!= 0:
            input= np.pad(input, ((0,0),(0,0),(padding, padding), (padding, padding)),
                                     'constant',constant_values = (0,0))
        num_kernels,_,filter_size,_ = kernel.shape
        # 计算输出特征图的宽度和高度
        output_W = (W + 2*padding - filter_size) // self.stride + 1
        output_W = int(output_W)
        output_H = (H + 2*padding - filter_size) // self.stride + 1
        output_H = int(output_H)

        # 初始化输出矩阵
        output_matrix = np.zeros((N, num_kernels, output_H, output_W))
        for h in range(output_H):
            for w in range(output_W):
                h_start = h * self.stride
                h_end = h_start + filter_size
                w_start = w * self.stride
                w_end = w_start + filter_size
                input_region = input[:, :, h_start:h_end, w_start:w_end].reshape((N, 1, C, filter_size, filter_size))
                output_matrix[:, :, h, w] += np.sum(input_region * kernel, axis=(2, 3, 4))
                if Bias is True:
                    output_matrix[:, :, h, w] += self.Bias
        return output_matrix
    
                    
    def forward(self, X):
        N,C,H,W = X.shape  
        # N for Batch, C for Channels, W for Width, H for Height
        assert(C == self.in_channels)
        self.input = X.copy()
        self.output = self.conv2d(X,self.Weight,self.padding)
        return self.output
    
    def backprop(self,grad):
        N, C, H, W = self.input.shape
        _, _, output_H, output_W = grad.shape
        reverse_kernel = self.Weight.transpose((1,0,2,3))
        reverse_kernel  = np.flip(reverse_kernel,axis=(2,3))
        grad_next =  self.conv2d(grad,reverse_kernel,self.filter_size-1,Bias=False)
        self.W_grad = np.zeros_like(self.Weight)
        for h in range(output_H):
            for w in range(output_W):
                tmp_back_grad = grad[:, :, h, w].T.reshape((self.out_channels, 1, 1, 1, N))
                tmp_x = self.input[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size].transpose((1, 2, 3, 0))
                self.W_grad += np.sum(tmp_back_grad * tmp_x, axis=4)

        self.B_grad = np.sum(grad, axis=(0, 2, 3))

        return grad_next
    
    
    def update_params(self, alpha):
        self.Weight -= alpha * self.W_grad
        self.Bias -= alpha * self.B_grad
        
class MaxPool:
    def __init__(self,pool_size=None):
        if pool_size is None:
            pool_size = 2
        self.pool_size = pool_size
        self.output = None
        self.input = None
        self.mask = None
    
    def forward(self, X):
        N,C,W,H = X.shape 
        self.input = X.copy()
        output_h = H // self.pool_size
        output_w = W // self.pool_size
        self.output = np.zeros((N,C,output_h,output_w))
        self.mask = np.zeros_like(X)  # 初始化最大值位置的掩码
        for i in range(output_h):
            for j in range(output_w):
                pool_window = X[:, :, i * self.pool_size:(i + 1) * self.pool_size, j * self.pool_size:(j + 1) * self.pool_size]
                self.mask[:, :, i * self.pool_size:(i + 1) * self.pool_size, j * self.pool_size:(j + 1) * self.pool_size] = (pool_window == np.max(pool_window, axis=(2, 3), keepdims=True))
                self.output[:,:,i,j] = np.max(X[:,:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size], axis=(2,3))
        return self.output
    
    def backprop(self, back_grad):
        # return back_grad[:, :, :, :, np.newaxis, np.newaxis] * self.mask
        N, C, H,W = back_grad.shape
        grad_next = np.zeros_like(self.input)

        for i in range(H):
            for j in range(W):
                # 获取在最大池化层前向传播时的最大值位置
                grad_next[:, :, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]= \
                self.mask[:,:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]* back_grad[:, :, i, j][:, :, None, None]

        return grad_next
        
class Linear:
    def __init__(self,input_size,output_size) :
        self.Weight = np.random.normal(scale=1, size=(input_size, output_size))
        self.W_grad = None
        self.Bias = np.zeros(output_size)
        self.B_grad = None
        self.input = None
        
    
    def forward(self,X):
        self.input = X.copy()
        return np.dot(X,self.Weight) + self.Bias
    
    def backprop(self,back_grad):
        self.W_grad = np.dot(self.input.T, back_grad)
        self.B_grad = np.sum(back_grad, axis=0)

        # 计算输入的梯度，用于传递给上一层
        grad_next = np.dot(back_grad, self.Weight.T)

        return grad_next
        
    
    def update_params(self, alpha):
        # 使用梯度下降更新权重和偏置
        self.Weight -= alpha * self.W_grad
        self.Bias -= alpha * self.B_grad
        
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
        #print(1,grad.sum())
        grad = self.Output.backprop(grad)
        #print(2,grad.sum())
        grad = self.Relu4.backprop(grad)
        #print(3,grad.sum())
        grad = self.Fc2.backprop(grad)
        #print(4,grad.sum())
        grad = self.Relu3.backprop(grad)
        #print(5,grad.sum())
        grad = self.Fc1.backprop(grad)
        #print(6,grad.sum())
        grad = grad.reshape(grad.shape[0],16,4,4)
        grad = self.Pool2.backprop(grad)
        #print(7,grad.sum())
        grad = self.Relu2.backprop(grad)
        #print(8,grad.sum())
        grad = self.Conv2.backprop(grad)
        #print(9,grad.sum())
        grad = self.Pool1.backprop(grad)
        #print(10,grad.sum())
        grad = self.Relu1.backprop(grad)
        #print(grad.sum())
        grad = self.Conv1.backprop(grad)
        #print(grad.sum())

    def update(self,alpha):
        self.Output.update_params(alpha)
        self.Fc2.update_params(alpha)
        self.Fc1.update_params(alpha)
        self.Conv2.update_params(alpha)
        self.Conv1.update_params(alpha)
    
    def get_params(self):
        return [self.Conv1.Weight, self.Conv1.Bias, self.Conv2.Weight, self.Conv2.Bias, self.Fc1.Weight, self.Fc1.Bias, 
                self.Fc2.Weight, self.Fc2.Bias, self.Output.Weight, self.Output.Bias]

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

