import numpy as np

class AdaGrad:
    def __init__(self, model, alpha=1e-3, beta1=0.9, beta2=0.999):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.model = model
        self.m = []
        self.v = []
        self.params = self.model.get_params()
        self.grad = None
        for param in self.params:
            self.m.append(np.zeros_like(param))
            self.v.append(np.zeros_like(param))
        

    def update(self):
        self.params = self.model.get_params()
        self.grad = self.model.get_grad()
        self.iter += 1
        #更新学习率
        alpha_t = self.alpha * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        #更新参数
        for i in range(len(self.params)):
            self.m[i] += (1 - self.beta1) * (self.grad[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (self.grad[i] ** 2 - self.v[i])
            self.params[i] -= alpha_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

