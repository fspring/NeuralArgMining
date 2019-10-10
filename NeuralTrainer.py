import numpy as np
import os
import copy

from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Layer, Dense, Embedding, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping

from crf import CRF
class NeuralTrainer:
    embedding_size = 300
    hidden_size = 100


    def __init__(self, maxlen, num_tags, word_index, embeddings, texts_to_eval_dir, dumpPath):
        self.sequences = []
        self.maxlen = maxlen
        self.vocab_size = len(word_index)+1
        self.num_tags = num_tags
        self.num_measures = 1 + 3*(num_tags - 1)
        self.word_index = word_index
        self.embeddings = embeddings
        self.texts_to_eval_dir = texts_to_eval_dir
        self.dumpPath = dumpPath
        self.model = None
        self.classes = ['']*num_tags
        self.read_tag_mapping()

    def read_tag_mapping(self):
        f = open('tag_mapping.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        tags = {}
        for mapping in lines:
            if(mapping == ''):
                continue
            map = mapping.split('\t')
            tags[int(map[1][0])-1] = map[0]
        for i in range(0, self.num_tags):
            self.classes[i] = tags[i]


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

        embedding_matrix = np.zeros((self.vocab_size, self.embedding_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def create_biLSTM(self, input):
        embeddingMatrix = self.createEmbeddings(self.word_index, self.embeddings)
        emb = Embedding(self.vocab_size, self.embedding_size, weights=[embeddingMatrix], input_length=self.maxlen,
                      trainable=False, mask_zero=True)(input)

        biLSTM_tensor = TimeDistributed(Dense(self.hidden_size, activation='relu'))(emb)
        biLSTM_tensor = Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh'))(biLSTM_tensor)
        biLSTM_tensor = Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh'))(biLSTM_tensor)

        return biLSTM_tensor

    def create_CRF(self, biLSTM_tensor):
        crf_tensor = TimeDistributed(Dense(20, activation='relu'))(biLSTM_tensor)

        crf = CRF(self.num_tags, sparse_target=False, learn_mode='join', test_mode='viterbi')
        crf_tensor = crf(crf_tensor)

        # self.crf_model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])

        return (crf_tensor, crf)

    def create_dist_layer(self, biLSTM_tensor):
        dist_tensor = TimeDistributed(Dense(1, activation='relu'))(biLSTM_tensor)
        # dist_tensor.compile(optimizer='adam', loss='mean_absolute_error', metrics='accuracy')

        return dist_tensor

    def create_model(self):
        input = Input(shape=(self.maxlen,))

        biLSTM_tensor = self.create_biLSTM(input)
        (crf_tensor, crf) = self.create_CRF(biLSTM_tensor)
        dist_tensor = self.create_dist_layer(biLSTM_tensor)

        self.model = Model(input=input, output=[crf_tensor,dist_tensor])
        # print(self.model.summary())
        self.model.compile(optimizer='adam', loss=[crf.loss_function,'mean_absolute_error'], metrics={'crf_1':[crf.accuracy], 'time_distributed_3':'accuracy'})

    def trainModel(self, x_train, y_train_class, y_train_dist, x_test, y_test_class, y_test_dist, unencodedY, testSet):
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

        y_train = [y_train_class,y_train_dist]
        self.model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor])

        y_test = [y_test_class,y_test_dist]
        scores = self.model.evaluate(x_test, y_test, batch_size=8, verbose=1)
        [y_pred_class, y_pred_dist] = self.model.predict(x_test)

        # scores: loss, crf_loss, dist_loss, crf_acc, dist_acc
        print("%s: %.2f%%" % (self.model.metrics_names[3], scores[3] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[4], scores[4] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[2], scores[2] * 100))

        self.write_evaluated_tests_to_file(x_test, y_pred_class, y_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath)
        spanEvalAt1 = self.spanEval(y_pred_class, y_pred_dist, unencodedY, 1.0)
        spanEvalAt075 = self.spanEval(y_pred_class, y_pred_dist, unencodedY, 0.75)
        spanEvalAt050 = self.spanEval(y_pred_class, y_pred_dist, unencodedY, 0.50)
        tagEval = self.tagEval(y_pred_class, unencodedY)
        # distEval = self.distEval(y_pred_dist,y_test_dist,unencodedY)

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
        cvscores = [0, [copy.deepcopy(empty_avg), copy.deepcopy(empty_avg), copy.deepcopy(empty_avg)], copy.deepcopy(empty_corr), copy.deepcopy(empty_corr), copy.deepcopy(empty_corr)]

        Y_class = Y[0] # Component identification
        Y_dist = Y[1] # Distances between related components
        for train, test in kfold.split(X, unencodedY):
            print('-------- Fold', foldNumber,'--------')
            X_train = []
            Y_train_class = []
            Y_train_dist = []
            for trainIndex in train:
                X_train.append(X[trainIndex])
                Y_train_class.append(Y_class[trainIndex])
                Y_train_dist.append(Y_dist[trainIndex])

            X_test = []
            Y_test_class = []
            Y_test_dist = []
            unencoded_Y = []
            for testIndex in test:
                X_test.append(X[testIndex])
                Y_test_class.append(Y_class[testIndex])
                Y_test_dist.append(Y_dist[trainIndex])
                unencoded_Y.append(unencodedY[testIndex])

            X_train = X_train + additionalX
            Y_train_class = Y_train_class + additionalY

            X_train = np.array(X_train)
            Y_train_class = np.array(Y_train_class)
            Y_train_dist = np.array(Y_train_dist)
            X_test = np.array(X_test)
            Y_test_class = np.array(Y_test_class)
            Y_test_dist = np.array(Y_test_dist)
            unencoded_Y = np.array(unencoded_Y)

            scores = self.trainModel(X_train, Y_train_class, Y_train_dist, X_test, Y_test_class,Y_test_dist, unencoded_Y, test)
            cvscores = self.handleScores(cvscores, scores, n_folds)
            foldNumber += 1
            
        print('Average results for the ten folds:')
        self.prettyPrintResults(cvscores)
        return cvscores

    def handleScores(self, oldScores, newScores, nFolds):
        newAccuracy = oldScores[0] + (newScores[0] / nFolds)
        empty = list(map(float,np.zeros(self.num_tags)))

        #[precision[class], recall[class], f1[class]]
        newTagScores = [copy.deepcopy(empty), copy.deepcopy(empty), copy.deepcopy(empty)]
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

    def tagEval(self, y_pred_class, unencodedY):
        i = 0
        precision = []
        recall = []
        f1 = []
        accuracy = []
        for result in y_pred_class:
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

        str_res = []
        float_res = []
        for i in range(0, self.num_tags):
            str_res.append(str(round(score_sum[i] / numTexts, 3)))
            float_res.append(round(score_sum[i] / numTexts, 4))

        print('\t'.join(self.classes))
        print('\t'.join(str_res))

        return float_res

    def distEval(self, y_pred_dist, y_test_dist, unencodedY):
        i = 0
        loss = []
        for result in y_pred_dist:
            sequenceLength = len(np.trim_zeros(unencodedY[i]))
            loss.append(mean_absolute_error(y_test_dist[i][:sequenceLength], result[:sequenceLength]))
            i += 1

        print("Loss = %.3f% (+/- %.3f%)" % (np.mean(loss), np.std(loss)))

    def prettyPrintResults(self, scores):

        #[acc, [precision[class], recall[class], f1[class]], --> avgs
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 100
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 75
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]]  --> at 50

        premise_index = self.classes.index('(I,premise)')
        claim_index = self.classes.index('(I,claim)')

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

        print('Precision')
        print('\t'.join(self.classes))
        print(str(round(scores[1][0][0], 3)) + '   ' + str(round(scores[1][0][1], 3)) + '     ' + str(
            round(scores[1][0][2], 3)))
        print('Recall')
        print('\t'.join(self.classes))
        print(str(round(scores[1][1][0], 3)) + '   ' + str(round(scores[1][1][1], 3)) + '     ' + str(
            round(scores[1][1][2], 3)))
        print('F1')
        print('\t'.join(self.classes))
        print(str(round(scores[1][2][0], 3)) + '   ' + str(round(scores[1][2][1], 3)) + '     ' + str(
            round(scores[1][2][2], 3)))

    def write_evaluated_tests_to_file(self, x_test, y_pred_class, y_pred_dist, testSet, text_dir, dumpPath):
        invword_index = {v: k for k, v in self.word_index.items()}
        texts = []

        for i in range(0,len(x_test)):
            text = []
            tags = y_pred_class[i]
            dists = y_pred_dist[i]
            j = 0
            for word in np.trim_zeros(x_test[i]):
                text.append([invword_index[word], np.argmax(tags[j]), dists[j]])
                j += 1
            texts.append(text)

        fileList = os.listdir(text_dir)
        filenames = []
        for file in fileList:
            fileName = dumpPath + '/' + file
            filenames.append(fileName)
        filenames = [filenames[x] for x in testSet]

        for i in range(0, len(texts)):
            textFile = open(filenames[i], "w", encoding='utf-8')
            for token in texts[i]:
                textFile.write(u'' + token[0] + ' ' + self.classes[token[1]] + ' ' + str(token[2]) + '\n')

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

    def spanEval(self, y_pred_class, y_pred_dist, unencodedY, threshold):
        goldSpans = self.spanCreator(unencodedY)
        empty = list(map(float,np.zeros(self.num_tags)))
        i = 0
        precision = copy.deepcopy(empty)
        recall = copy.deepcopy(empty)
        f1 = copy.deepcopy(empty)
        predictedSpanTypes = copy.deepcopy(empty)
        goldSpanTypes = copy.deepcopy(empty)
        precisionCorrectSpans = copy.deepcopy(empty)
        recallCorrectSpans = copy.deepcopy(empty)
        for result in y_pred_class:
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

        precision_arg = 0
        goldSpanTypes_arg = 0
        for i in range(1,self.num_tags):
            precision_arg += precisionCorrectSpans[i]
            goldSpanTypes_arg += goldSpanTypes[i]
        accuracy = precision_arg / goldSpanTypes_arg

        for i in range(0, self.num_tags):
            if (predictedSpanTypes[i] != 0):
                precision[i] = (precisionCorrectSpans[i] / predictedSpanTypes[i])
            if (goldSpanTypes[i] != 0):
                recall[i] = (recallCorrectSpans[i] / goldSpanTypes[i])
            if ((precision[i] + recall[i]) != 0):
                f1[i] = 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i]))

        premise_index = self.classes.index('(I,premise)')
        claim_index = self.classes.index('(I,claim)')
        print('Accuracy at ' + str(threshold) + ' - ' + str(round(accuracy, 3)))
        print('Precision for premises at ' + str(threshold) + ' - ' + str(round(precision[premise_index], 3)))
        print('Precision for claims at ' + str(threshold) + ' - ' + str(round(precision[claim_index], 3)))
        print('Recall for premises at ' + str(threshold) + ' - ' + str(round(recall[premise_index], 3)))
        print('Recall for claims at ' + str(threshold) + ' - ' + str(round(recall[claim_index], 3)))
        print('F1 for premises at ' + str(threshold) + ' - ' + str(round(f1[premise_index], 3)))
        print('F1 for claims at ' + str(threshold) + ' - ' + str(round(f1[claim_index], 3)))

        ret = [round(accuracy, 4),round(precision[premise_index], 4),round(precision[claim_index], 4),round(recall[premise_index], 4),
            round(recall[claim_index], 4),round(f1[premise_index], 4),round(f1[claim_index], 4)]
        return ret
