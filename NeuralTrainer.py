import numpy as np
import os

from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Layer, Dense, Embedding, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping

from crf import CRF
class NeuralTrainer:
    embedding_size = 300
    hidden_size = 100

    def __init__(self, maxlen, num_tags, wordIndex, embeddings, textsToEval, dumpPath, rel):
        self.sequences = []
        self.maxlen = maxlen
        self.max_features = len(wordIndex)
        self.num_tags = num_tags
        self.num_measures = 1 + 3*(num_tags - 1)
        self.wordIndex = wordIndex
        self.embeddings = embeddings
        self.textsToEval = textsToEval
        self.dumpPath = dumpPath
        self.rel = rel

    def decodeTags(self, tags):
        newtags = []
        for tag in tags:
            newtag = np.argmax(tag)
            newtags.append(newtag)
        return newtags

    def createEmbeddings(self, word_index, embeddings):
        embeddings_index = {}
        path = 'Embeddings/' + embeddings + '.txt'
        f = open(path, "r", encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((self.max_features + 1, self.embedding_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def createModel(self):
        embeddingMatrix = self.createEmbeddings(self.wordIndex, self.embeddings)
        model = Sequential()

        model.add(
            Embedding(self.max_features + 1, self.embedding_size, weights=[embeddingMatrix], input_length=self.maxlen,
                      trainable=False, mask_zero=True))

        model.add(TimeDistributed(Dense(self.hidden_size, activation='relu')))
        model.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh')))
        model.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh')))
        model.add(TimeDistributed(Dense(20, activation='relu')))

        crf = CRF(self.num_tags, sparse_target=False, learn_mode='join', test_mode='viterbi')
        model.add(crf)
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])

        return model

    def trainModel(self, x_train, y_train, x_test, y_test, unencodedY, testSet):
        model = self.createModel()
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
        model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor])
        scores = model.evaluate(x_test, y_test, batch_size=8, verbose=1)
        y_pred = model.predict(x_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        self.printEvaluatedTexts(x_test, y_pred, testSet, self.textsToEval, self.dumpPath)
        spanEvalAt1 = self.spanEval(y_pred, unencodedY, 1.0)
        spanEvalAt075 = self.spanEval(y_pred, unencodedY, 0.75)
        spanEvalAt050 = self.spanEval(y_pred, unencodedY, 0.50)
        tagEval = self.tagEval(y_pred, unencodedY)

        return [scores[1], tagEval, spanEvalAt1, spanEvalAt075, spanEvalAt050]

    def crossValidate(self, X, Y, additionalX, additionalY, unencodedY):
        seed = 42
        n_folds = 10
        # n_folds = 2
        foldNumber  = 1

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        #[acc, [precision[class], recall[class], f1[class]],[100_correct],[75_correct],[50_correct]]
        #percent_correct: [acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]
        empty_avg = list(map(float,np.zeros(self.num_tags)))
        # nr_measures = 1 + 3*(self.num_tags-1)
        empty_corr = list(map(float,np.zeros(self.num_measures)))
        cvscores = [0, [empty_avg, empty_avg, empty_avg], empty_corr, empty_corr, empty_corr]

        for train, test in kfold.split(X, unencodedY):
            print('Fold')
            print(foldNumber)
            xTrainSet = []
            yTrainSet = []
            xTestSet = []
            yTestSet = []
            unencodedYSet = []
            for trainIndex in train:
                xTrainSet.append(X[trainIndex])
                yTrainSet.append(Y[trainIndex])
            for testIndex in test:
                xTestSet.append(X[testIndex])
                yTestSet.append(Y[testIndex])
                unencodedYSet.append(unencodedY[testIndex])

            xTrainSet = xTrainSet + additionalX
            yTrainSet = yTrainSet + additionalY
            scores = self.trainModel(xTrainSet, yTrainSet, xTestSet, yTestSet, unencodedYSet, test)
            cvscores = self.handleScores(cvscores, scores, n_folds)
            foldNumber += 1
        print('Average results for the ten folds:')
        self.prettyPrintResults(cvscores)
        return cvscores

    def handleScores(self, oldScores, newScores, nFolds):
        newAccuracy = oldScores[0] + (newScores[0] / nFolds)
        empty = list(map(float,np.zeros(self.num_tags)))

        #[precision[class], recall[class], f1[class]]
        newTagScores = [empty, empty, empty]
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]
        # nr_measures = 1 + 3*(self.num_tags-1)

        newSpanAt1Scores = list(map(float,np.zeros(self.num_measures)))
        newSpanAt075Scores = list(map(float,np.zeros(self.num_measures)))
        newSpanAt050Scores = list(map(float,np.zeros(self.num_measures)))

        for i in range(0, 3):
            # for j in range(0, self.num_tags):
                # newTagScores[i][j] = oldScores[1][i][j] + (newScores[1][i][j] / nFolds)
            newTagScores[i] = list(map(float,np.array(oldScores[1][i]) + (np.array(newScores[1][i]) / nFolds)))
        for j in range(0, self.num_measures):
            newSpanAt1Scores[j] = oldScores[2][j] + (newScores[2][j] / nFolds)
        for j in range(0, self.num_measures):
            newSpanAt075Scores[j] = oldScores[3][j] + (newScores[3][j] / nFolds)
        for j in range(0, self.num_measures):
            newSpanAt050Scores[j] = oldScores[4][j] + (newScores[4][j] / nFolds)

            #[acc, [precision[class], recall[class], f1[class]], --> avgs
            #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 100
            #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 75
            #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]]  --> at 50
        return [newAccuracy, newTagScores, newSpanAt1Scores, newSpanAt075Scores, newSpanAt050Scores]

    def tagEval(self, y_pred, unencodedY, ):
        i = 0
        precision = []
        recall = []
        f1 = []
        accuracy = []
        for result in y_pred:
            sequenceLength = len(np.trim_zeros(unencodedY[i]))
            result = np.resize(result, (sequenceLength, self.num_tags))
            classes = np.argmax(result, axis=1)
            accuracy.append(accuracy_score(np.trim_zeros(unencodedY[i]), np.add(classes, 1)))
            scores = precision_recall_fscore_support(np.trim_zeros(unencodedY[i]), np.add(classes, 1))
            precision.append(np.pad(scores[0], (0,(self.num_tags - len(scores[0]))), 'constant'))
            recall.append(np.pad(scores[1], (0,(self.num_tags - len(scores[0]))), 'constant'))
            f1.append(np.pad(scores[2], (0,(self.num_tags - len(scores[0]))), 'constant'))
            i += 1
        print("Accuracy = %.3f%% (+/- %.3f%%)" % (np.mean(accuracy), np.std(accuracy)))
        precision = self.prettyPrintScore(precision, 'Precision')
        recall = self.prettyPrintScore(recall, 'Recall')
        f1 = self.prettyPrintScore(f1, 'F1')

        return [precision, recall, f1]

    def prettyPrintScore(self, score, scoreName):
        print(scoreName)
        numTexts = len(score)
        score_sum = np.zeros(self.num_tags)
        for scoreValue in score:
            score_sum += scoreValue[:self.num_tags]

        if(self.rel):
            print('(o,|,|)\t(i,claim,0)\t(i,premise,d)')
            print(str(round(score_sum[0] / numTexts, 3)) + '\t' + str(
                round(score_sum[1] / numTexts, 3)) + '\t' + str(np.round(score_sum[2:] / numTexts, 3)))
        else:
            print('(i,premise)' + '(o,|)   ' + '(i,claim)')
            print(str(round(score_sum[0] / numTexts, 3)) + '   ' + str(
                round(score_sum[1] / numTexts, 3)) + '     ' + str(np.round(score_sum[2] / numTexts, 3)))

        # return [round(score_sum[0] / numTexts, 4), round(score_sum[1] / numTexts, 4),
        #         round(score_sum[2:] / numTexts, 4)]
        return list(map(float,np.round(score_sum/numTexts,4)))

    def prettyPrintResults(self, scores):

        #[acc, [precision[class], recall[class], f1[class]], --> avgs
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 100
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 75
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]]  --> at 50

        print('Accuracy - ' + str(round(scores[0], 4)))

        print('Accuracy at ' + str(1) + ' - ' + str(round(scores[2][0], 3)))
        print('Accuracy at ' + str(0.75) + ' - ' + str(round(scores[3][0], 3)))
        print('Accuracy at ' + str(0.5) + ' - ' + str(round(scores[4][0], 3)))

        print('Precision for premises at ' + str(1) + ' - ' + str(round(scores[2][1], 3)))
        print('Precision for claims at ' + str(1) + ' - ' + str(round(scores[2][2], 3)))
        print('Precision for premises at ' + str(0.75) + ' - ' + str(round(scores[3][1], 3)))
        print('Precision for claims at ' + str(0.75) + ' - ' + str(round(scores[3][2], 3)))
        print('Precision for premises at ' + str(0.5) + ' - ' + str(round(scores[4][1], 3)))
        print('Precision for claims at ' + str(0.5) + ' - ' + str(round(scores[4][2], 3)))

        print('Recall for premises at ' + str(1) + ' - ' + str(round(scores[2][3], 3)))
        print('Recall for claims at ' + str(1) + ' - ' + str(round(scores[2][4], 3)))
        print('Recall for premises at ' + str(0.75) + ' - ' + str(round(scores[3][3], 3)))
        print('Recall for claims at ' + str(0.75) + ' - ' + str(round(scores[3][4], 3)))
        print('Recall for premises at ' + str(0.5) + ' - ' + str(round(scores[4][3], 3)))
        print('Recall for claims at ' + str(0.5) + ' - ' + str(round(scores[4][4], 3)))

        print('F1 for premises at ' + str(1) + ' - ' + str(round(scores[2][5], 3)))
        print('F1 for claims at ' + str(1) + ' - ' + str(round(scores[2][6], 3)))
        print('F1 for premises at ' + str(0.75) + ' - ' + str(round(scores[3][5], 3)))
        print('F1 for claims at ' + str(0.75) + ' - ' + str(round(scores[3][6], 3)))
        print('F1 for premises at ' + str(0.5) + ' - ' + str(round(scores[4][5], 3)))
        print('F1 for claims at ' + str(0.5) + ' - ' + str(round(scores[4][6], 3)))

        print('@prettyPrintResults: scores[1] -- ', str(scores[1]))

        print('Precision')
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        # print(str(round(score_sum[0] / numTexts, 3)) + '   ' + str(
        #     round(score_sum[1] / numTexts, 3)) + '     ' + str(np.round(score_sum[2:] / numTexts, 3)))
        print(str(round(scores[1][0][0], 3)) + '   ' + str(round(scores[1][0][1], 3)) + '     ' + str(
            round(scores[1][0][2], 3)))
        print('Recall')
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        print(str(round(scores[1][1][0], 3)) + '   ' + str(round(scores[1][1][1], 3)) + '     ' + str(
            round(scores[1][1][2], 3)))
        print('F1')
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        print(str(round(scores[1][2][0], 3)) + '   ' + str(round(scores[1][2][1], 3)) + '     ' + str(
            round(scores[1][2][2], 3)))

    def printEvaluatedTexts(self, x_test, y_pred, testSet, textList, dumpPath):
        invWordIndex = {v: k for k, v in self.wordIndex.items()}
        texts = []

        for i in range(0,len(x_test)):
            text = []
            tags = y_pred[i]
            j = 0
            for word in np.trim_zeros(x_test[i]):
                text.append([invWordIndex[word], np.argmax(tags[j])])
                j += 1
            texts.append(text)

        fileList = os.listdir(textList)
        filenames = []
        for file in fileList:
            fileName = dumpPath + '/' + file
            filenames.append(fileName)
        filenames = [filenames[x] for x in testSet]

        if(self.rel):
            classes = ['(O,|,|)', '(I,Claim,0)', '(I,Premise,-1)', '(I,Premise,-2)', '(I,Premise,-3)', '(I,Premise,-4)', '(I,Premise,-5)',
                        '(I,Premise,0)', '(I,Premise,-6)', '(I,Premise,-7)', '(I,Premise,-8)', '(I,Premise,-9)', '(I,Premise,-10)',
                        '(I,Premise,-11)', '(I,Premise,-12)']
            for i in range(0, len(texts)):
                textFile = open(filenames[i], "w", encoding='utf-8')
                for token in texts[i]:
                    textFile.write(u'' + token[0] + ' ' + classes[token[1]] + '\n')
        else:
            classes = ['(I,Premise)', '(O,|)', '(I,Claim)']
            for i in range(0, len(texts)):
                textFile = open(filenames[i], "w", encoding='utf-8')
                for token in texts[i]:
                    textFile.write(u'' + token[0] + ' ' + classes[token[1]] + '\n')

    def spanCreator(self, unencodedY):
        spans = []
        for text in unencodedY:
            text = np.trim_zeros(text)
            textSpans = {}
            startPosition = 0
            currentPosition = 0
            lastTag = text[0]
            for tag in text:
                if tag != lastTag:
                    endPosition = currentPosition - 1
                    textSpans[startPosition] = endPosition
                    startPosition = currentPosition
                lastTag = tag
                currentPosition += 1
            endPosition = currentPosition - 1
            textSpans[startPosition] = endPosition
            spans.append(textSpans)

        return spans

    def trimSpans(self, predictedSpanStart, predictedSpanEnd, goldSpans):
        trimmedSpans = {}
        for goldSpanStart, goldSpanEnd in goldSpans[0].items():
            if ((predictedSpanStart >= goldSpanStart) and (predictedSpanEnd <= goldSpanEnd)):
                trimmedSpans[predictedSpanStart] = predictedSpanEnd
        return trimmedSpans

    def spanEval(self, y_pred, unencodedY, threshold):
        goldSpans = self.spanCreator(unencodedY)
        empty = list(map(float,np.zeros(self.num_tags)))
        i = 0
        precision = empty
        recall = empty
        f1 = empty
        predictedSpanTypes = empty
        goldSpanTypes = empty
        precisionCorrectSpans = empty
        recallCorrectSpans = empty
        for result in y_pred:
            sequenceLength = len(np.trim_zeros(unencodedY[i]))
            result = np.resize(result, (sequenceLength, self.num_tags))
            classes = np.argmax(result, axis=1)
            classes = np.add(classes, 1)

            for spanStart, spanEnd in goldSpans[i].items():
                goldSpanTypes[unencodedY[i][spanStart] - 1] += 1

            for spanStart, spanEnd in goldSpans[i].items():
                predicted = classes[spanStart:spanEnd + 1]
                possibleSpans = self.spanCreator([predicted])

                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    predictedSpanTypes[classes[spanStart + possibleSpanStart] - 1] += 1
                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    if (((possibleSpanEnd - possibleSpanStart + 1) >= ((spanEnd - spanStart + 1) * threshold))
                            and (classes[spanStart + possibleSpanStart] == unencodedY[i][
                                spanStart + possibleSpanStart])):
                        precisionCorrectSpans[classes[spanStart + possibleSpanStart] - 1] += 1
                        break
                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    if (((possibleSpanEnd - possibleSpanStart + 1) >= ((spanEnd - spanStart + 1) * threshold))
                            and (classes[spanStart + possibleSpanStart] == unencodedY[i][
                                spanStart + possibleSpanStart])):
                        recallCorrectSpans[classes[spanStart + possibleSpanStart] - 1] += 1
            i += 1

        if(self.rel):
            precision_arg = 0
            goldSpanTypes_arg = 0
            for i in range(1,self.num_tags):
                precision_arg += precisionCorrectSpans[i]
                goldSpanTypes_arg += goldSpanTypes[i]
            accuracy = precision_arg / goldSpanTypes_arg
        else:
            accuracy = ((precisionCorrectSpans[0] + precisionCorrectSpans[2]) / (goldSpanTypes[0] + goldSpanTypes[2]))

        for i in range(0, self.num_tags):
            if (predictedSpanTypes[i] != 0):
                precision[i] = (precisionCorrectSpans[i] / predictedSpanTypes[i])
            if (goldSpanTypes[i] != 0):
                recall[i] = (recallCorrectSpans[i] / goldSpanTypes[i])
            if ((precision[i] + recall[i]) != 0):
                f1[i] = 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i]))

        if(self.rel):
            precision_premises = np.array(precision[2:])
            recall_premises = np.array(recall[2:])
            f1_premises = np.array(f1[2:])

            print('Accuracy at ' + str(threshold) + ' - ' + str(round(accuracy, 3)))
            print('Precision for premises at ' + str(threshold) + ' - ' + str(np.round(precision_premises, 3)))
            print('Precision for claims at ' + str(threshold) + ' - ' + str(round(precision[1], 3)))
            print('Recall for premises at ' + str(threshold) + ' - ' + str(np.round(recall_premises, 3)))
            print('Recall for claims at ' + str(threshold) + ' - ' + str(round(recall[1], 3)))
            print('F1 for premises at ' + str(threshold) + ' - ' + str(np.round(f1_premises, 3)))
            print('F1 for claims at ' + str(threshold) + ' - ' + str(round(f1[1], 3)))

            ret = [round(accuracy, 4)]
            for value in precision_premises:
                ret.append(value)
            ret.append(precision[1])
            for value in recall_premises:
                ret.append(value)
            ret.append(recall[1])
            for value in f1_premises:
                ret.append(value)
            ret.append(f1[1])

        else:

            print('Accuracy at ' + str(threshold) + ' - ' + str(round(accuracy, 3)))
            print('Precision for premises at ' + str(threshold) + ' - ' + str(round(precision[0], 3)))
            print('Precision for claims at ' + str(threshold) + ' - ' + str(round(precision[2], 3)))
            print('Recall for premises at ' + str(threshold) + ' - ' + str(round(recall[0], 3)))
            print('Recall for claims at ' + str(threshold) + ' - ' + str(round(recall[2], 3)))
            print('F1 for premises at ' + str(threshold) + ' - ' + str(round(f1[0], 3)))
            print('F1 for claims at ' + str(threshold) + ' - ' + str(round(f1[2], 3)))

            ret = [round(accuracy, 4), round(precision_premises, 4), round(precision_claims, 4), round(recall_premises, 4),
                    round(recall_claims, 4), round(f1_premises, 4), round(f1_claims, 4)]
        print('ret: ',len(ret))
        return ret
