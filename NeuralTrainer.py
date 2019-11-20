import numpy as np
import os
import pathlib
import copy

from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Layer, Dense, Activation, Embedding, LSTM, Bidirectional, Lambda, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping

from keras.losses import mean_absolute_error

import tensorflow as tf

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

class SoftArgMax:
    def __init__(self):
        self.layer = None

    def soft_argmax_func(self, x, beta=1e10):
        x = tf.split(x, [3, 1], -1)
        x_range = tf.range(x[0].shape.as_list()[-1], dtype=x[0].dtype)
        output = tf.reduce_sum(tf.nn.softmax(x[0]*beta) * x_range, axis=-1)

        lower = K.constant(0.5)
        upper = K.constant(1.5)
        zero = K.zeros_like(x[1])

        return K.switch(K.all(K.stack([K.greater_equal(output, lower), K.less(output, upper)], axis=0), axis=0), zero, x[1])

    def create_soft_argmax_layer(self):
        self.layer = Lambda(self.soft_argmax_func, output_shape=(1,), name='lambda_softargmax')

    def loss_func(self, y_true, y_pred):
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        y_pred = K.squeeze(y_pred, axis=-1)
        zero = K.constant(0)

        return K.switch(K.equal(y_pred, zero), K.zeros_like(mae), mae)


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
        self.tags = ['']*num_tags
        self.arg_classes = ['']*num_tags
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
            self.tags[i] = tags[i]
            name = tags[i].split(',')[-1][:-1]
            self.arg_classes[i] = name


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

    def create_CRF(self, biLSTM_tensor, learn, test):
        crf_tensor = TimeDistributed(Dense(20, activation='relu'))(biLSTM_tensor)

        crf = CRF(self.num_tags, sparse_target=False, learn_mode=learn, test_mode=test, name='crf_layer')

        crf_tensor = crf(crf_tensor)

        return crf_tensor

    def create_dist_layer(self, biLSTM_tensor, crf_tensor):
        dist_tensor = TimeDistributed(Dense(1, activation='relu'), name='distance_layer')(biLSTM_tensor)

        soft_argmax = SoftArgMax()
        soft_argmax.create_soft_argmax_layer()

        concat = concatenate([crf_tensor, dist_tensor], axis=-1)

        output = TimeDistributed(soft_argmax.layer, name='softargmax')(concat)

        return (output, soft_argmax)

    def create_model(self):
        input = Input(shape=(self.maxlen,))

        biLSTM_tensor = self.create_biLSTM(input)
        crf_tensor = self.create_CRF(biLSTM_tensor, 'marginal', 'marginal')

        (dist_tensor, soft_argmax) = self.create_dist_layer(biLSTM_tensor, crf_tensor)

        self.model = Model(input=input, output=[crf_tensor,dist_tensor])
        # print(self.model.summary())

        # self.model.compile(optimizer='adam', loss=[crf_loss,soft_argmax.loss_func], metrics={'crf_layer':[crf_accuracy], 'softargmax':'mae'})
        self.model.compile(optimizer='adam', loss=[crf_loss,'mean_absolute_error'], metrics={'crf_layer':[crf_accuracy], 'softargmax':'mae'})

    def create_baseline_model(self):
        input = Input(shape=(self.maxlen,))

        biLSTM_tensor = self.create_biLSTM(input)
        crf_tensor = self.create_CRF(biLSTM_tensor, 'join', 'viterbi')

        self.model = Model(input=input, output=crf_tensor)
        # print(self.model.summary())

        self.model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_accuracy])

    ## for each premise find closest claim ahead
    def predict_baseline_distances_claim(self, pred_tags):
        pred_dists = []
        for i in range(0, len(pred_tags)):
            file_dists = []
            text_size = len(pred_tags[i])
            for j in range(0, text_size):
                arg_class = np.argmax(pred_tags[i][j])
                dist = 0
                if arg_class == 0:
                    for k in range(j+1, text_size):
                        arg_rel = np.argmax(pred_tags[i][k])
                        dist += 1
                        if arg_rel == 2:
                            break
                file_dists.append([dist])
            pred_dists.append(file_dists)
        return pred_dists

    ## for each premise find closest arg_component ahead
    def predict_baseline_distances_next(self, pred_tags):
        pred_dists = []
        for i in range(0, len(pred_tags)):
            file_dists = []
            text_size = len(pred_tags[i])
            for j in range(0, text_size):
                arg_class = np.argmax(pred_tags[i][j])
                dist = 0
                if arg_class == 0:
                    k = j+1
                    while k < text_size and np.argmax(pred_tags[i][k]) == 0:
                        dist += 1
                        k += 1
                    for n in range(k, text_size):
                        arg_rel = np.argmax(pred_tags[i][n])
                        if arg_rel == 1:
                            dist += 1
                        elif arg_rel == 2:
                            dist += 1
                            break
                file_dists.append([dist])
            pred_dists.append(file_dists)
        return pred_dists

    ## for each premise find closest claim ahead and for each claim find closest premise ahead
    def predict_baseline_distances_all_next(self, pred_tags):
        pred_dists = []
        for i in range(0, len(pred_tags)):
            file_dists = []
            text_size = len(pred_tags[i])
            for j in range(0, text_size):
                arg_class = np.argmax(pred_tags[i][j])
                dist = 0
                if arg_class == 0:
                    k = j+1
                    while k < text_size and np.argmax(pred_tags[i][k]) == 0:
                        dist += 1
                        k += 1
                    for n in range(k, text_size):
                        arg_rel = np.argmax(pred_tags[i][n])
                        if arg_rel == 1:
                            dist += 1
                        elif arg_rel == 2:
                            dist += 1
                            break
                elif arg_class == 2:
                    k = j+1
                    while k < text_size and np.argmax(pred_tags[i][k]) == 2:
                        dist += 1
                        k += 1
                    for n in range(k, text_size):
                        arg_rel = np.argmax(pred_tags[i][n])
                        if arg_rel == 1:
                            dist += 1
                        elif arg_rel == 0:
                            dist += 1
                            break
                file_dists.append([dist])
            pred_dists.append(file_dists)
        return pred_dists

    def train_baseline_model(self, x_train, y_train, x_test, y_test_class, y_test_dist, unencodedY, testSet):
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

        self.model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor])

        scores = self.model.evaluate(x_test, y_test_class, batch_size=8, verbose=1)
        y_pred_class = self.model.predict(x_test)

        # scores: loss, crf_acc
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

        b_pred_dist = self.predict_baseline_distances_claim(y_pred_class)

        self.write_evaluated_tests_to_file(x_test, y_pred_class, b_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath + '_baseline')
        spanEvalAt1 = self.spanEval(y_pred_class, b_pred_dist, unencodedY, 1.0)
        spanEvalAt075 = self.spanEval(y_pred_class, b_pred_dist, unencodedY, 0.75)
        spanEvalAt050 = self.spanEval(y_pred_class, b_pred_dist, unencodedY, 0.50)
        tagEval = self.tagEval(y_pred_class, unencodedY)

        print('------- Distances Baseline -------')
        dist_eval = self.dist_eval(b_pred_dist, y_test_dist, y_test_class, unencodedY)
        edge_eval = self.edge_eval(b_pred_dist, y_test_dist, y_test_class, unencodedY)

        return [[scores[1], tagEval, spanEvalAt1, spanEvalAt075, spanEvalAt050], dist_eval, edge_eval]

    def correct_dist_prediction(self, arg_pred, dist_pred, unencodedY):
        print('=========== CORRECTING ===========')
        f = open('correction_debug.txt', 'w')
        for i in range(0, len(dist_pred)):
            text_size = len(np.trim_zeros(unencodedY[i]))
            for j in range(0, text_size): #ensure dist points to first token in arg comp or zero
                src_arg = np.argmax(arg_pred[i][j])
                pred_dist = int(round(dist_pred[i][j][0]))
                if src_arg == 1: #non-arg
                    dist_pred[i][j][0] = 0
                elif src_arg == 0: #premise
                    if pred_dist == 0:
                        dist_pred[i][j][0] = 0
                        continue
                    tgt_index = j + pred_dist
                    if (tgt_index >= text_size) or (np.argmax(arg_pred[i][tgt_index]) != 2):
                        dist_pred[i][j][0] = 0 #does not point to claim
                        continue
                    while np.argmax(arg_pred[i][tgt_index - 1]) == 2: #not first claim token
                        tgt_index -= 1
                        if tgt_index == j:
                            print('fuck me up fam')
                            break
                    dist_pred[i][j][0] = tgt_index - j
                elif src_arg == 2: #claim
                    if pred_dist == 0:
                        dist_pred[i][j][0] = 0
                        continue
                    tgt_index = j + pred_dist
                    if (tgt_index >= text_size) or (np.argmax(arg_pred[i][tgt_index]) != 0):
                        dist_pred[i][j][0] = 0 #does not point to premise
                        continue
                    while np.argmax(arg_pred[i][tgt_index - 1]) == 0: #not first premise token
                        tgt_index -= 1
                        if tgt_index == j:
                            print('fuck me up fam squared')
                            break
                    dist_pred[i][j][0] = tgt_index - j
            f.write(u'i: ' + str(i) + ' - phase 1: ' + str(dist_pred[i]) + '\n')
            k = 0
            while k < text_size: #ensure uniformity: all tokens in src arg comp point to same tgt token
                src_orig = k
                src_arg = np.argmax(arg_pred[i][k])
                pred_dist = dist_pred[i][k][0]
                if src_arg == 1:
                    k += 1
                    continue
                if pred_dist == 0:
                    tgt_freq = {'none': 1}
                else:
                    tgt_freq = {pred_dist: 1}

                m =  k + 1
                while (m < text_size) and (np.argmax(arg_pred[i][m]) == src_arg):
                    pred_dist = dist_pred[i][m][0]
                    if pred_dist == 0:
                        tgt_orig = 'none'
                    else:
                        tgt_orig = pred_dist + (m - src_orig)

                    if tgt_orig in tgt_freq.keys():
                        tgt_freq[tgt_orig] += 1
                    else:
                        tgt_freq[tgt_orig] = 1
                    m += 1
                k = m

                max_value = max(tgt_freq.values()) #get most common decision
                most_freq = []
                for dist in tgt_freq.keys():
                    if tgt_freq[dist] == max_value:
                        most_freq.append(dist)
                if len(most_freq) > 1:
                    most_freq = [0] #decides none
                    # most_freq = [min(most_freq)] #decides closest
                if most_freq[0] == 'none' or most_freq[0] == 0:
                    for l in range(src_orig, k):
                        dist_pred[i][l] = [0]
                        continue
                else:
                    for l in range(src_orig, k):
                        dist_pred[i][l] = most_freq
                        most_freq[0] -= 1

            f.write(u'i: ' + str(i) + ' - phase 2: ' + str(dist_pred[i]) + '\n')
        f.close()
        print('=========== DONE ===========')
        return dist_pred


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

        self.write_evaluated_tests_to_file(x_test, y_pred_class, y_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath + '_BCorr')

        c_pred_dist = self.correct_dist_prediction(y_pred_class, y_pred_dist, unencodedY)

        self.write_evaluated_tests_to_file(x_test, y_pred_class, c_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath)
        spanEvalAt1 = self.spanEval(y_pred_class, y_pred_dist, unencodedY, 1.0)
        spanEvalAt075 = self.spanEval(y_pred_class, y_pred_dist, unencodedY, 0.75)
        spanEvalAt050 = self.spanEval(y_pred_class, y_pred_dist, unencodedY, 0.50)
        tagEval = self.tagEval(y_pred_class, unencodedY)

        print('------- Distances from model -------')
        dist_eval = self.dist_eval(c_pred_dist, y_test_dist, y_test_class, unencodedY)
        edge_eval = self.edge_eval(c_pred_dist, y_test_dist, y_test_class, unencodedY)

        return [[scores[1], tagEval, spanEvalAt1, spanEvalAt075, spanEvalAt050], dist_eval, edge_eval]

    def crossValidate(self, X, Y, additionalX, additionalY, unencodedY):
        seed = 42
        n_folds = 10
        foldNumber  = 1

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        csv_header = 'Fold'
        for i in range(0, self.num_tags):
            tag = self.tags[i].split(',')
            arg_type = tag[-1][:-1]
            csv_header += ',' + arg_type + ',Total ' + arg_type
        csv_header += '\n'
        csv_entries_dist = ''
        csv_entries_edge = ''

        #[acc, [precision[class], recall[class], f1[class]],[100_correct],[75_correct],[50_correct]]
        #percent_correct: [acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]
        empty_avg = list(map(float,np.zeros(self.num_tags)))
        # nr_measures = 1 + 3*(self.num_tags-1)
        empty_corr = list(map(float,np.zeros(self.num_measures)))
        cvscores = [0, [copy.deepcopy(empty_avg), copy.deepcopy(empty_avg), copy.deepcopy(empty_avg)],
                        copy.deepcopy(empty_corr), copy.deepcopy(empty_corr), copy.deepcopy(empty_corr)]

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
                Y_test_dist.append(Y_dist[testIndex])
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

            # scores = self.trainModel(X_train, Y_train_class, Y_train_dist, X_test, Y_test_class,Y_test_dist, unencoded_Y, test)
            scores = self.train_baseline_model(X_train, Y_train_class, X_test, Y_test_class,Y_test_dist, unencoded_Y, test)
            cvscores = self.handleScores(cvscores, scores[0], n_folds)
            csv_entries_dist = self.distance_stats_to_csv(scores[1], foldNumber, csv_entries_dist)
            csv_entries_edge = self.distance_stats_to_csv(scores[2], foldNumber, csv_entries_edge)
            foldNumber += 1

        print('Average results for the ten folds:')
        self.prettyPrintResults(cvscores)
        ## write distance prediction stats to csv
        f = open('model_dist_predictions_ften.csv', 'w')
        f.write(csv_header + csv_entries_dist)
        f.close()

        f = open('model_edge_predictions_ften.csv', 'w')
        f.write(csv_header + csv_entries_edge)
        f.close()

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

        print('\t'.join(self.tags))
        print('\t'.join(str_res))

        return float_res

    def dist_eval(self, y_pred_dist, y_test_dist, y_test_class, unencodedY):
        nr_files = len(y_test_dist)
        tp = [0]*self.num_tags
        n = [0]*self.num_tags
        for i in range(0, nr_files):
            text_size = len(np.trim_zeros(unencodedY[i]))
            for j in range(0, text_size):
                pred = round(y_pred_dist[i][j][0])
                tag = np.argmax(y_test_class[i][j])
                n[tag] += 1
                if int(pred) == int(y_test_dist[i][j][0]):
                    tp[tag] += 1
        for i in range(0, self.num_tags):
            print('======', self.tags[i], '======')
            print('Correct distances:', tp[i], '------- Ratio', tp[i]/n[i], '\n')
        return (tp, n)

    def edge_eval(self, y_pred_dist, y_test_dist, y_test_class, unencodedY):
        nr_files = len(y_test_dist)
        tp = [0]*self.num_tags
        n = [0]*self.num_tags
        for i in range(0, nr_files):
            text_size = len(np.trim_zeros(unencodedY[i]))
            for j in range(0, text_size):
                pred = int(round(y_pred_dist[i][j][0]))
                true = int(y_test_dist[i][j][0])
                tag = np.argmax(y_test_class[i][j])
                n[tag] += 1

                pred_dest = j + pred
                true_dest_start = j + true
                true_dest_end = true_dest_start + 1
                if pred == true:
                    tp[tag] += 1
                elif (pred_dest >= text_size) or (true_dest_end >= text_size):
                    continue
                else:
                    dest_tag = np.argmax(y_test_class[i][true_dest_start])
                    while (true_dest_end < text_size) and (dest_tag == np.argmax(y_test_class[i][true_dest_end])):
                        true_dest_end += 1
                    if (pred_dest > true_dest_start) and (pred_dest < true_dest_end):
                        tp[tag] += 1

        for i in range(0, self.num_tags):
            print('======', self.tags[i], '======')
            print('Correct distances:', tp[i], '------- Ratio', tp[i]/n[i], '\n')
        return (tp, n)

    def distance_stats_to_csv(self, predictions, fold, entries):
        entries += str(fold)
        for i in range(0, self.num_tags):
            entries += ',' + str(predictions[0][i]) + ',' + str(predictions[1][i])
        entries += '\n'

        return entries

    def prettyPrintResults(self, scores):

        #[acc, [precision[class], recall[class], f1[class]], --> avgs
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 100
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 75
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]]  --> at 50

        premise_index = self.tags.index('(I,premise)')
        claim_index = self.tags.index('(I,claim)')

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
        print('\t'.join(self.tags))
        print(str(round(scores[1][0][0], 3)) + '   ' + str(round(scores[1][0][1], 3)) + '     ' + str(
            round(scores[1][0][2], 3)))
        print('Recall')
        print('\t'.join(self.tags))
        print(str(round(scores[1][1][0], 3)) + '   ' + str(round(scores[1][1][1], 3)) + '     ' + str(
            round(scores[1][1][2], 3)))
        print('F1')
        print('\t'.join(self.tags))
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
                textFile.write(u'' + token[0] + ' ' + self.tags[token[1]] + ' ' + '{:8.3f}'.format(token[2][0]) + '\n')

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

        premise_index = self.tags.index('(I,premise)')
        claim_index = self.tags.index('(I,claim)')
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
