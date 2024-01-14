import numpy as np

class AdaGrad:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.model = model
        self.m = None
        self.v = None
        self.params = None
        self.grad = None

    def step(self):
        self.params = self.model.get_params()
        self.grad = self.model.get_grad()
        if self.m is None:
            self.m, self.v = [], []
            for param in self.params:
                self.m.append(np.zeros_like(param))
            for g in self.grad:
                self.v.append(np.zeros_like(g))
            assert(len(self.m) == len(self.params))
            assert(len(self.v) == len(self.grad))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(self.params)):
            self.m[i] += (1 - self.beta1) * (self.grad[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (self.grad[i] ** 2 - self.v[i])
            self.params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

