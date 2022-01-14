import csv
import numpy


def load_tweets(file):
    tweets = []
    hate_speech = 0
    bad_words = 0
    neither = 0

    with open(file) as csv_file:
        next(csv_file)
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for line in csv_reader:
            assert int(line[1]) == int(line[2]) + int(line[3]) + int(line[4])
            doc = {
                'id':                   int(line[0]),
                'classification':       numpy.array([
                    float(line[2])/float(line[1]),
                    float(line[3])/float(line[1]),
                    float(line[4])/float(line[1])
                ], dtype=float),
                'tweet':                line[6],
                'word_embeddings':      [],
                'interesting_words':    [],
                'word_avg_embedding':   None
            }
            tweets.append(doc)

            if line[5] == '0':
                hate_speech += 1
            elif line[5] == '1':
                bad_words += 1
            elif line[5] == '2':
                neither += 1
    return tweets, [hate_speech, bad_words, neither]
