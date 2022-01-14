import torch

class NN_MOD(torch.nn.Module):
    def __init__(self, inputSize, outputSize, learningRate, hiddenSize, vocabSize):
        torch.manual_seed(1)
        super(NN_MOD, self).__init__()
        self.hiddenSize = hiddenSize
        self.word_embeddings = torch.nn.Embedding(vocabSize, inputSize)
        self.lstm = torch.nn.LSTM(inputSize, hiddenSize, bidirectional=False)
        self.classifier = torch.nn.Linear(hiddenSize, outputSize)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        tag_space = self.classifier(lstm_out.view(len(sentence), -1))
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores
