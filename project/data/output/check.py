import csv
import os
import optparse

def load_results(file):
    testResults = []

    with open(file) as csv_file:
        next(csv_file)
        csv_reader = csv.reader(csv_file, delimeter=',', quotechar='"')
        for line in csv_reader:
            doc = {
                'tweet' = line[0],
                'classification' = line[1],
                'prediction' = line[2]
            }
            testResults.append(doc)
    
    return testResults

def eval(testResults, cat):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for (_, answer, prediction) in testResults:
        if cat == prediction and cat == answer:
            tp += 1
        elif cat != prediction and cat != answer:
            tn += 1
        elif cat == prediction and cat != answer:
            fp += 1
        elif cat != prediction and cat == answer:
            fn += 1
        else:
            raise Exception('Unexpected answer/prediction')

    acc = float((tp + tn) / max(1, (tp + tn + fp + fn))*100)
    preP = float((tp) / max(1, (tp + fp))*100)
    recP = float((tp) / max(1, (tp + fn))*100)
    fScore = float(2*preP*recP / max(1, (preP+recP)))

    return (
        acc,
        preP,
        recP,
        fScore
    )

def printEval(testResults):
    print(f"Run Complete\nConfiguration:\n\tNeural Network: {opts.neural_network}\n\tFeatures: {opts.features}\n\tLabels: {opts.labels}\n\tLearning Rate: {opts.learning_rate}\n\tHidden Layers: {opts.hidden_layers}\n\tEpochs: {opts.epochs}\n\tTraining Partition: {opts.training_partition*100.0}%")

    (a, p, r, f) = eval(testResults, 0)
    print("Hate Speech")
    print(f"\tAccuracy      : {a}%")
    print(f"\tPrecision(+)  : {p}%")
    print(f"\tRecall(+)     : {r}%")
    print(f"\tF-Score       : {f}%")

    (a, p, r, f) = eval(testResults, 1)
    print("Bad Words")
    print(f"\tAccuracy      : {a}%")
    print(f"\tPrecision(+)  : {p}%")
    print(f"\tRecall(+)     : {r}%")
    print(f"\tF-Score       : {f}%")

    (a, p, r, f) = eval(testResults, 2)
    print("Safe Tweet")
    print(f"\tAccuracy      : {a}%")
    print(f"\tPrecision(+)  : {p}%")
    print(f"\tRecall(+)     : {r}%")
    print(f"\tF-Score       : {f}%")

if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option('-f' '--file', dest='file', default=os.path.join('results.txt'))

    testResults = load_results(opts.file)
    printEval(testResults)