import numpy as np

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
        self.Weight = np.random.normal(scale=1, size=(out_channels,in_channels,filter_size[0],filter_size[1]))
        #第一个维度是输出通道数，对应着有这么多个卷积核；第二个维度是输入通道数，这是因为卷积核需要负责讲in个channel的输入变成out个channel的输出

        self.W_grad = None
        self.Bias = np.zeros(out_channels) #每个卷积核有一个偏置参数
        self.B_grad = None
    
    def Conv2D(self):
        # def conv(x,y,i,j): #从(x,y)到(x+filter_size,t+filter_size)的input与weight[i,j]的卷积计算
        #     return np.sum(X[x:x+self.filter_size,y:y+self.filter_size] * self.Weight[i,j])
        
        # res = []
        # for i in range(self.in_channels):
        #     for o in range(self.out_channels):
        #         res.extend(joblib.Parallel(n_jobs=-1,verbose=0)(joblib.delayed(conv)(h*self.stride,w*self.stride,i,o) for h in range(self.output_H) for w in range(self.output_W) ))
        # return np.array(res).reshape(self.in_channels,self.out_channels,self.output_H,self.output_W)
        for i in range(self.in_channels):
            for o in range(self.out_channels):
                for h in range(self.output_H):
                    for w in range(self.output_W):
                        self.output[i,o,h,w] = np.sum(self.input[h*self.stride:h*self.stride+self.filter_size[0],
                                                                 w*self.stride:w*self.stride+self.filter_size[1]] * self.Weight[i,o])
                
    def forward(self, X):
        C,W,H = X.shape  
        # C for Channels, W for Width, H for Height

        # 进行padding操作，填充0
        if self.padding!= 0:
            for channels in range(C):
                X[channels] = np.pad(X[channels], ((self.padding, self.padding), (self.padding, self.padding)),
                                     'constant',constant_values = (0,0))
            
        output_W = (W + 2*self.padding - self.filter_size[0]) // self.stride + 1
        output_W = int(output_W)
        output_H = (H + 2*self.padding - self.filter_size[1]) // self.stride + 1
        output_H = int(output_H)
        self.output = np.zeros((self.out_channels, output_H, output_W))
        
        for o in range(self.out_channels):
                for h in range(output_H):
                    for w in range(output_W):
                        self.output[o,h,w] = np.sum(X[:,h*self.stride:h*self.stride+self.filter_size[0],
                                                                 w*self.stride:w*self.stride+self.filter_size[1]] * self.Weight[o])+self.Bias[o]
        return self.output
    
    def backprop(self, dout):
        pass

class MaxPool:
    def __init__(self,pool_size=None,stride = 1):
        if pool_size is None:
            pool_size = [2, 2]
        self.pool_size = pool_size
        self.stride = stride
        self.output = None
    
    def forward(self, X):
        in_channels,h,w = X.shape
        output_h = h // self.pool_size[0]
        output_w = w // self.pool_size[1]
        self.output = np.zeros((in_channels,output_h,output_w))
        for i in range(output_h):
            for j in range(output_w):
                self.output[:,i,j] = np.max(X[:, i*self.pool_size[0]:(i+1)*self.pool_size[0], j*self.pool_size[1]:(j+1)*self.pool_size[1]], axis=(1, 2))
        return self.output
    
    def backprop(self, back_grad):
        pass

class Linear:
    def __init__(self,input_size,output_size) :
        self.Weight = np.random.normal(scale=1, size=(input_size, output_size))
        self.W_grad = None
        self.Bias = np.zeros(output_size)
        self.B_grad = None
    
    def forward(self,X):
        return np.dot(X,self.Weight) + self.Bias
    
    def back_prop(self,back_grad):
        pass
        