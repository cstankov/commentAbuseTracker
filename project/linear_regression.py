import torch


class NN_MOD(torch.nn.Module):
    def __init__(self, inputSize, outputSize, learningRate, hiddenLayers):
        super(NN_MOD, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)

    def forward(self, x):
        out = self.linear(x)
        return out
