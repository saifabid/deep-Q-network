import torch.nn as nn
class FCNet(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(num_inputs, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, num_outputs),
                                nn.Softmax(dim=0)) 
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_loss_function(self, loss_f):
        self.loss_f = loss_f

    def forward(self, x):
        return self.net(x)
    
    def predict(self,x):
        return self.forward(x)
    
    def train(self, xs, ys):
        self.optimizer.zero_grad()
        predicted_ys = self.net(xs)
        loss_v = self.loss_f(predicted_ys, ys)
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()