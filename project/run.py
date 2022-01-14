import optparse
import os
import re
import pymagnitude
import numpy
import torch
from torch.autograd import Variable
from load_tweets import load_tweets
import linear_regression
import deep_neural_network
import matplotlib.pyplot as plt


def getGloveEmbeddings(glove):
    glove = pymagnitude.Magnitude(glove)
    gloveModel = {}
    for word, vector in glove:
        gloveModel[word] = vector
    return gloveModel


def findWordEmbeddings(tweets, glove):
    for tweet in tweets:
        for Word in tweet['tweet'].split():
            word = re.sub(r'\W+', '', Word.lower())
            if word in glove:
                tweet['word_embeddings'].append((word, glove[word]))
            else:
                tweet['interesting_words'].append(word)


def averageWordEmbeddings(tweets, vLength=100):
    for tweet in tweets:
        nEmbeddings = len(tweet['word_embeddings'])
        tweet['word_avg_embedding'] = numpy.zeros(vLength, dtype=numpy.float)
        for _, embedding in tweet['word_embeddings']:
            tweet['word_avg_embedding'] += embedding / nEmbeddings


def wordProbabilities(tweets):
    pW = {}
    totalWords = 0
    for tweet in tweets:
        for word in tweet['tweet'].split():
            totalWords += 1
            if word not in pW:
                pW[word] = 1
            else:
                pW[word] += 1
    for word, numOccurrences in pW.items():
        pW[word] = numOccurrences/totalWords
    return pW


def weightedAverageWordEmbeddings(tweets, vLength=100, a=0.0001):
    pW = wordProbabilities(tweets)
    for tweet in tweets:
        nEmbeddings = len(tweet['word_embeddings'])
        tweet['word_avg_embedding'] = numpy.zeros(vLength, dtype=numpy.float)
        for word, embedding in tweet['word_embeddings']:
            if word in pW:
                pOfWord = pW[word]
            else:
                pOfWord = 0

            rareWeight = a / (a + pOfWord)
            tweet['word_avg_embedding'] += (rareWeight *
                                            embedding) / nEmbeddings


def partitionTweets(tweets, totals, split):
    assert split > 0.0 and split < 1.0
    nTweets = len(tweets)
    trainSize = int(nTweets * split)

    trainingTweets = []
    testingTweets = []
    repReq = [int(total * split) for total in totals]

    def appendTweet(counterIndex, tweet):
        if repReq[counterIndex] > 0:
            repReq[counterIndex] -= 1
            trainingTweets.append(tweet)
        else:
            testingTweets.append(tweet)

    for tweet in tweets:
        classification = int(torch.argmax(
            torch.from_numpy(tweet['classification'])))
        assert classification >= 0 and classification < 3
        appendTweet(classification, tweet)

    return (trainingTweets, testingTweets)


def trainNeuralModel(PKG, opts, trainingTweets):
    model = PKG.NN_MOD(opts.features, opts.labels,
                       opts.learning_rate, opts.hidden_layers)

    criterion = model.criterion
    optimizer = model.optimizer

    for epoch in range(opts.epochs):
        for tweet in trainingTweets:
            input = Variable(torch.from_numpy(tweet['word_avg_embedding']))
            label = Variable(torch.from_numpy(tweet['classification']))

            optimizer.zero_grad()
            outputs = model(input.float())
            loss = criterion(outputs, label.float())

            loss.backward()
            optimizer.step()
    return model


def testNeuralModel(model, testingTweets):
    results = []
    xs = []
    ys = []
    zs = []
    pC = []
    cC = []

    def colorPickCat(output):
        idx = torch.argmax(output)
        if idx == 0:
            return 'r'
        elif idx == 1:
            return 'y'
        else:
            return 'g'

    with torch.no_grad():
        for tweet in testingTweets:
            input = Variable(torch.from_numpy(tweet['word_avg_embedding']))
            predicted = model(input.float())
            results.append(
                (
                    tweet['tweet'],
                    torch.argmax(torch.from_numpy(tweet['classification'])),
                    torch.argmax(predicted)
                )
            )
            xs.append(predicted[0])
            ys.append(predicted[1])
            zs.append(predicted[2])
            cC.append(colorPickCat(torch.from_numpy(tweet['classification'])))
            pC.append(colorPickCat(predicted))

    return results, (xs, ys, zs, cC, pC)


def outputResults(testResults, file):
    classEnum = ['hate_speech', 'bad_words', 'safe']
    with open(file, 'w') as output_file:
        for tweet, answer, prediction in testResults:
            output_file.write(
                f"\"\"\"{tweet}\"\"\",{classEnum[answer]},{classEnum[prediction]}\n")


def PLOT(plotInfo):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(plotInfo[0], plotInfo[1], plotInfo[2], color=plotInfo[3])

    ax.set_xlabel('Hate Speech')
    ax.set_ylabel('Bad Words')
    ax.set_zlabel('Safe Tweet')

    plt.show()


def run(opts):
    # setup tweet/sentence embeddings
    tweets, totals = load_tweets(opts.input)
    glove = getGloveEmbeddings(opts.glove)
    findWordEmbeddings(tweets, glove)
    weightedAverageWordEmbeddings(tweets)

    # partition training/testing data sets
    (trainingTweets, testingTweets) = partitionTweets(
        tweets, totals, opts.training_partition)

    # select neural network class
    if 'linear_regression' in opts.neural_network:
        PKG = linear_regression
    elif 'deep_neural_network' in opts.neural_network:
        PKG = deep_neural_network
    else:
        raise ValueError(f"{opts.neural_network} alogrithm not recognized")

    # train and test
    trainedModel = trainNeuralModel(PKG, opts, trainingTweets)
    testResults, plotInfo = testNeuralModel(trainedModel, testingTweets)

    outputResults(testResults, os.path.join('data', 'output', opts.output))
    return testResults


if __name__ == "__main__":
    optparser = optparse.OptionParser()

    # dependent files
    optparser.add_option('-i', '--input', dest='input',
                         default=os.path.join('data', 'tweets.csv'))
    optparser.add_option('-g', '--glove', dest='glove',
                         default=os.path.join('data', 'glove.6B.100d.magnitude'))
    optparser.add_option('-o', '--output', dest='output',
                         default='results.txt')

    # select from neural network class
    # 1. deep_neural_network
    # 2. linear_regression
    # 3. tbd...
    optparser.add_option('-n', '--neural_network', dest='neural_network',
                         default='linear_regression')

    # options less likely to be changed
    optparser.add_option('--features', dest='features', default=100)
    optparser.add_option('--labels', dest='labels', default=3)

    # options that may be changed/experimented with
    optparser.add_option('--learning_rate',
                         dest='learning_rate', type="float", default=0.1)
    optparser.add_option('--hidden_layers',
                         dest='hidden_layers', type="int", default=10)
    optparser.add_option('--epochs', dest='epochs', type="int", default=10)
    optparser.add_option('--training_partition',
                         dest='training_partition', type="float", default=0.2)

    (opts, _) = optparser.parse_args()

    testResults = run(opts)
