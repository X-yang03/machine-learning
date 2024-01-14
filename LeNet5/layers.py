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
        self.Weight = np.random.normal(scale=1e-3, size=(out_channels,in_channels,filter_size,filter_size))
        #第一个维度是输出通道数，对应着有这么多个卷积核；第二个维度是输入通道数，这是因为卷积核需要负责讲in个channel的输入变成out个channel的输出
        self.W_grad = np.zeros_like(self.Weight)
        self.Bias = np.zeros(out_channels) #每个卷积核有一个偏置参数
        self.B_grad = np.zeros_like(self.Bias)
    
    def conv2d(self,input,kernel,padding = 0,Bias = True): #卷积运算,使用kernel对input进行卷积
        N,C,H,W = input.shape #batchsize,channels,Height,Width
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
        #进行卷积运算
        for h in range(output_H):
            for w in range(output_W):
                #计算卷积的区域(h_start,h_end,w_start,w_end):
                h_start = h * self.stride       
                h_end = h_start + filter_size   
                w_start = w * self.stride       
                w_end = w_start + filter_size
                #选取input的这一块区域，并进行reshape
                input_region = input[:, :, h_start:h_end, w_start:w_end].reshape((N, 1, C, filter_size, filter_size))
                output_matrix[:, :, h, w] += np.sum(input_region * kernel, axis=(2, 3, 4)) #通过numpy矩阵运算进行卷积
                if Bias is True: #是否加上偏置参数Bias
                    output_matrix[:, :, h, w] += self.Bias
        return output_matrix
    
                    
    def forward(self, X):
        N,C,H,W = X.shape  
        # N for Batch, C for Channels, W for Width, H for Height
        assert(C == self.in_channels)
        self.input = X.copy()
        if self.padding!= 0:
            self.input= np.pad(self.input, ((0,0),(0,0),(self.padding, self.padding), (self.padding, self.padding)),
                                     'constant',constant_values = (0,0))
        self.output = self.conv2d(self.input,self.Weight)
        return self.output
    
    def backprop(self,grad):
        N, C, H, W = self.input.shape
        _, _, output_H, output_W = grad.shape
        #得到卷积核的180°翻转
        reverse_kernel = self.Weight.transpose((1,0,2,3)) 
        reverse_kernel  = np.flip(reverse_kernel,axis=(2,3))
        #卷积层输入的梯度实际上是输出的梯度经过padding后，与180°翻转的卷积核进行卷积的结果
        grad_next =  self.conv2d(grad,reverse_kernel,self.filter_size-1,Bias=False)
        #计算卷积核的梯度
        self.W_grad = np.zeros_like(self.Weight)
        for h in range(output_H):
            for w in range(output_W):
                tmp_back_grad = grad[:, :, h, w].T.reshape((self.out_channels, 1, 1, 1, N))
                tmp_x = self.input[:, :, h * self.stride:h * self.stride + self.filter_size, w * self.stride:w * self.stride + self.filter_size].transpose((1, 2, 3, 0))
                self.W_grad += np.sum(tmp_back_grad * tmp_x, axis=4)

        self.B_grad = np.sum(grad, axis=(0, 2, 3))  #计算偏置参数的梯度

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
        self.mask = np.zeros_like(X)  # 初始化最大值位置的掩码,用于记录每个2*2块中最大项的位置
        for i in range(output_h):
            for j in range(output_w):
                pool_window = X[:, :, i * self.pool_size:(i + 1) * self.pool_size, j * self.pool_size:(j + 1) * self.pool_size]
                self.mask[:, :, i * self.pool_size:(i + 1) * self.pool_size, j * self.pool_size:(j + 1) * self.pool_size] \
                    = (pool_window == np.max(pool_window, axis=(2, 3), keepdims=True)) # 记录最大值位置
                
                #取得最大值
                self.output[:,:,i,j] = np.max(X[:,:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size], axis=(2,3))
        return self.output
    
    def backprop(self, back_grad):
        N, C, H,W = back_grad.shape
        grad_next = np.zeros_like(self.input)

        for i in range(H):
            for j in range(W):
                #根据mask，将每一个梯度映射到每个2*2块的最大值位置
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
        #利用numpy的矩阵乘法，计算输出
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