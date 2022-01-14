import torch


class NN_MOD(torch.nn.Module):
    def __init__(self, inputSize, outputSize, learningRate, hiddenSize):
        super(NN_MOD, self).__init__()
        self.hidden = torch.nn.Linear(inputSize, hiddenSize)
        self.predict = torch.nn.Linear(hiddenSize, outputSize)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.predict(x)
        return x
