import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import real_data_target
from torch.distributions.multivariate_normal import *

def noise(quantity, size, rho, gen_from_marked):
    if gen_from_marked:
        sigma_P = torch.tensor([[rho,0], [0,1/rho]])
        m = MultivariateNormal(torch.zeros(2), sigma_P)
        n = Variable(m.sample([size,50]))
        n = n.reshape(size,100)
        n = Variable(torch.randn(size, 100))
        return n
    else:
        return Variable(torch.randn(quantity, size))
    

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, out_features, rho=2, gen_from_marked = False, leakyRelu=0.2):
        super(GeneratorNet, self).__init__()
        
        self.gen_from_marked = gen_from_marked
        self.rho = rho
        self.in_features = 100
        self.layers = [128, 128, 128]
        self.layers.insert(0, self.in_features)

        for count in range(0, len(self.layers)-1):
            self.add_module("hidden_" + str(count), 
                nn.Sequential(
                    nn.Linear(self.layers[count], self.layers[count+1]),
                    nn.LeakyReLU(leakyRelu)
                )
            )

        
        self.add_module("out", 
            nn.Sequential(
                nn.Linear(self.layers[-1], out_features)
            )
        )
    
    def forward(self, x):
        for name, module in self.named_children():
            x = module(x)
        return x

    def create_data(self, quantity):
        points = noise(quantity, self.in_features, self.rho, self.gen_from_marked)
        try:
            data=self.forward(points.cuda())
        except:
            data=self.forward(points.cpu())
        return data.detach().numpy()

def train_generator(optimizer, discriminator, loss, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error