import numpy as np
import os
import pathlib

import Evaluator as ev
import RelationBaseline as rb
import PostProcessing as pp

from sklearn.model_selection import KFold

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
        x = tf.split(x, [4, 1], -1)
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
        self.word_index = word_index
        self.embeddings = embeddings
        self.texts_to_eval_dir = texts_to_eval_dir
        self.dumpPath = dumpPath
        self.model = None
        self.tags = ['']*num_tags
        self.arg_classes = ['']*num_tags
        self.transition_matrix = None
        self.save_weights = False
        self.read_tag_mapping()
        self.set_transition_matrix()
        num_measures = 1 + 3*(num_tags - 2)
        self.evaluator = ev.Evaluator(self.num_tags, num_measures, self.tags)
        self.postprocessing = pp.PostProcessing(self.num_tags, self.tags)

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
            if tags[i] == '(O)':
                self.arg_classes[i] = '|'
            elif tags[i] == '(P)':
                self.arg_classes[i] = 'premise'
            elif tags[i] == '(C)':
                self.arg_classes[i] = 'claim'
            elif tags[i] == '(I)':
                self.arg_classes[i] = 'inside'

    def set_transition_matrix(self):
        transition_matrix = np.array([[1]*self.num_tags]*self.num_tags)
        # matrix is initialized to 1
        # this function sets some entries to -1
        for i in range(0, self.num_tags):
            if self.tags[i] == '(O)':
                for j in range(0, self.num_tags):
                    if self.tags[j] == '(P)': # impossible transition to (O)
                        transition_matrix[i][j] = -1
                    elif self.tags[j] == '(C)': # impossible transition to (O)
                        transition_matrix[i][j] = -1
            elif self.tags[i] == '(P)':
                for j in range(0, self.num_tags):
                    if self.tags[j] == '(P)': # impossible transition to (P)
                        transition_matrix[i][j] = -1
                    elif self.tags[j] == '(C)': # impossible transition to (P)
                        transition_matrix[i][j] = -1
            elif self.tags[i] == '(C)':
                for j in range(0, self.num_tags):
                    if self.tags[j] == '(P)': # impossible transition to (C)
                        transition_matrix[i][j] = -1
                    elif self.tags[j] == '(C)': # impossible transition to (C)
                        transition_matrix[i][j] = -1
            elif self.tags[i] == '(I)':
                for j in range(0, self.num_tags):
                    if self.tags[j] == '(O)': # impossible transition to (I)
                        transition_matrix[i][j] = -1
        print(transition_matrix) #debug
        self.transition_matrix = transition_matrix

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
                      trainable=False, mask_zero=True, name='embedding')(input)

        biLSTM_tensor = TimeDistributed(Dense(self.hidden_size, activation='relu'), name='time_distributed_1')(emb)
        biLSTM_tensor = Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh'), name='biLSTM_1')(biLSTM_tensor)
        biLSTM_tensor = Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh'), name='biLSTM_2')(biLSTM_tensor)

        return biLSTM_tensor

    def create_CRF(self, biLSTM_tensor, learn, test):
        crf_tensor = TimeDistributed(Dense(20, activation='relu'), name='time_distributed_2')(biLSTM_tensor)

        chain_matrix = keras.initializers.Constant(self.transition_matrix)

        if learn == 'marginal': #loaded model or std CRF-dist model
            crf = CRF(self.num_tags, sparse_target=False, learn_mode=learn, test_mode=test,
                    chain_initializer=chain_matrix, name='crf_layer')
            # crf = CRF(self.num_tags, sparse_target=False, learn_mode=learn, test_mode=test, name='crf_layer')

        else: #baseline model
            crf = CRF(self.num_tags, sparse_target=False, learn_mode=learn, test_mode=test, name='crf_layer')

        crf_tensor = crf(crf_tensor)

        return crf_tensor

    def create_dist_layer(self, biLSTM_tensor, crf_tensor):
        dist_tensor = TimeDistributed(Dense(1, activation='relu'), name='distance_layer')(biLSTM_tensor)

        soft_argmax = SoftArgMax()
        soft_argmax.create_soft_argmax_layer()

        concat = concatenate([crf_tensor, dist_tensor], axis=-1, name='concatenate')

        output = TimeDistributed(soft_argmax.layer, name='softargmax')(concat)

        return (output, soft_argmax)

    def create_model(self):
        input = Input(shape=(self.maxlen,), name='input')

        biLSTM_tensor = self.create_biLSTM(input)
        crf_tensor = self.create_CRF(biLSTM_tensor, 'marginal', 'marginal')

        (dist_tensor, soft_argmax) = self.create_dist_layer(biLSTM_tensor, crf_tensor)

        self.model = Model(input=input, output=[crf_tensor,dist_tensor])
        print(self.model.summary()) #debug

        #loss_weights=[1.0, 0.10],
        self.model.compile(optimizer='adam', loss=[crf_loss,'mean_absolute_error'], loss_weights=[1.0, 0.10], metrics={'crf_layer':[crf_accuracy], 'softargmax':'mae'})

        if self.save_weights:
            print('MODEL LOADED FROM FILE')
            # self.model.load_weights('baseline_weights.h5', by_name=True)
            
            base_crf_tensor = self.create_CRF(biLSTM_tensor, 'join', 'viterbi')
            baseline_model = Model(input=input, output=base_crf_tensor)
            print(baseline_model.summary()) #debug
            baseline_model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_accuracy])

            baseline_model.load_weights('baseline_weights.h5', by_name=True)

            base_layers = baseline_model.layers
            model_layers = self.model.layers
            for i in range(0, len(base_layers)):
                print(model_layers[i].name, base_layers[i].name)
                assert model_layers[i].name == base_layers[i].name
                layer_name = base_layers[i].name
                self.model.get_layer(layer_name).set_weights(base_layers[i].get_weights())
                # self.model.get_layer(layer_name).trainable = False
            



        # self.model.compile(optimizer='adam', loss=[crf_loss,soft_argmax.loss_func], metrics={'crf_layer':[crf_accuracy], 'softargmax':'mae'})

    def create_baseline_model(self, type):
        input = Input(shape=(self.maxlen,))

        biLSTM_tensor = self.create_biLSTM(input)
        # if type == 'dual':
        #     crf_tensor = self.create_CRF(biLSTM_tensor, 'marginal', 'marginal')
        # else:
        #     crf_tensor = self.create_CRF(biLSTM_tensor, 'join', 'viterbi')
        crf_tensor = self.create_CRF(biLSTM_tensor, 'join', 'viterbi')

        self.model = Model(input=input, output=crf_tensor)
        print(self.model.summary()) #debug

        self.model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_accuracy])

    def train_baseline_model(self, x_train, y_train, x_test, y_test_class, y_test_dist, unencodedY, testSet):
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
        print('x train shape:', x_train.shape, 'x test shape:', x_test.shape)
        print('y train shape:', y_train.shape, 'y test shape:', y_test_class.shape)

        self.model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor])

        scores = self.model.evaluate(x_test, y_test_class, batch_size=8, verbose=1)
        y_pred_class = self.model.predict(x_test)

        # scores: loss, crf_acc
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

        (y_pred_class, b_pred_dist) = rb.predict_baseline_distances_claim(y_pred_class)

        if len(y_pred_class) != len(b_pred_dist):
            print('not same nr files')
        else:
            for i in range(0, len(y_pred_class)):
                if len(y_pred_class[i]) != len(b_pred_dist[i]):
                    print(i, 'not same file length')

        if not os.path.exists(self.dumpPath + '_baseline'):
            os.makedirs(self.dumpPath + '_baseline')
        self.write_evaluated_tests_to_file(x_test, y_pred_class, b_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath + '_baseline')

        (true_spans, pred_spans) = self.postprocessing.replace_argument_tag(y_pred_class, unencodedY)

        spanEvalAt1 = self.evaluator.spanEval(pred_spans, true_spans, 1.0)
        spanEvalAt075 = self.evaluator.spanEval(pred_spans, true_spans, 0.75)
        spanEvalAt050 = self.evaluator.spanEval(pred_spans, true_spans, 0.50)
        tagEval = self.evaluator.tagEval(pred_spans, true_spans)

        print('------- Distances Baseline -------')
        dist_eval = self.evaluator.dist_eval(pred_spans, true_spans, b_pred_dist, y_test_dist)

        return [scores[1], tagEval, spanEvalAt1, spanEvalAt075, spanEvalAt050, dist_eval]

    def trainModel(self, x_train, y_train_class, y_train_dist, x_test, y_test_class, y_test_dist, unencodedY, testSet):
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

        y_train = [y_train_class,y_train_dist]
        self.model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor])

        layer = self.model.get_layer('crf_layer')
        weights = layer.get_weights()
        gen_matrix = weights[1]

        # print('before:', gen_matrix) #debug

        for i in range(0, self.num_tags):
            for j in range(0, self.num_tags):
                if self.transition_matrix[i][j] == 1:
                    gen_matrix[i][j] = 1

        # print('after:', gen_matrix) #debug
        weights[1] = gen_matrix

        self.model.get_layer('crf_layer').set_weights(weights)
        # print('check:', self.model.get_layer('crf_layer').get_weights()[1]) #debug

        y_test = [y_test_class,y_test_dist]
        scores = self.model.evaluate(x_test, y_test, batch_size=8, verbose=1)
        [y_pred_class, y_pred_dist] = self.model.predict(x_test)

        # scores: loss, crf_loss, dist_loss, crf_acc, dist_acc
        print("%s: %.2f%%" % (self.model.metrics_names[3], scores[3] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[4], scores[4] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[2], scores[2] * 100))
        if not os.path.exists(self.dumpPath + '_BCorr'):
            os.makedirs(self.dumpPath + '_BCorr')
        self.write_evaluated_tests_to_file(x_test, y_pred_class, y_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath + '_BCorr')

        (y_pred_class, c_pred_dist) = self.postprocessing.correct_dist_prediction(y_pred_class, y_pred_dist, unencodedY)

        self.write_evaluated_tests_to_file(x_test, y_pred_class, c_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath)

        (true_spans, pred_spans) = self.postprocessing.replace_argument_tag(y_pred_class, unencodedY)

        spanEvalAt1 = self.evaluator.spanEval(pred_spans, true_spans, 1.0)
        spanEvalAt075 = self.evaluator.spanEval(pred_spans, true_spans, 0.75)
        spanEvalAt050 = self.evaluator.spanEval(pred_spans, true_spans, 0.50)
        tagEval = self.evaluator.tagEval(pred_spans, true_spans)

        print('------- Distances from model -------')
        dist_eval = self.evaluator.dist_eval(pred_spans, true_spans, c_pred_dist, y_test_dist)

        return [scores[1], tagEval, spanEvalAt1, spanEvalAt075, spanEvalAt050, dist_eval]

    def crossValidate(self, X, Y, additionalX, additionalY, unencodedY, model_type):
        seed = 42
        n_folds = 10
        foldNumber  = 1

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        csv_header = 'Fold'
        for i in range(1, self.num_tags):
            tag = self.arg_classes[i]
            csv_header += ',' + tag + ',Total ' + tag
        csv_header += '\n'
        csv_entries_dist = ''
        csv_entries_edge = ''

        cvscores = self.evaluator.empty_cvscores()


        print(np.array(X).shape)
        print(np.array(Y[0]).shape, np.array(Y[1]).shape)

        Y_class = Y[0] # Component identification
        Y_dist = Y[1] # Distances between related components
        additionalY_class = additionalY[0]
        additionalY_dist = additionalY[1]
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
            Y_train_class = Y_train_class + additionalY_class
            Y_train_dist = Y_train_dist + additionalY_dist

            X_train = np.array(X_train)
            Y_train_class = np.array(Y_train_class)
            Y_train_dist = np.array(Y_train_dist)
            X_test = np.array(X_test)
            Y_test_class = np.array(Y_test_class)
            Y_test_dist = np.array(Y_test_dist)
            unencoded_Y = np.array(unencoded_Y)

            print('after additional x shape:', X_train.shape)
            print('after additional y class shape:', Y_train_class.shape)
            print('after additional y dist shape:', Y_train_dist.shape)
            print('test x shape:', X_test.shape)
            print('test y class index:', Y_test_class.shape)
            print('test y dist index:', Y_test_dist.shape)

            scores = None
            if model_type == 'baseline':
                scores = self.train_baseline_model(X_train, Y_train_class, X_test, Y_test_class,Y_test_dist, unencoded_Y, test)
            elif model_type == 'crf_dist':
                scores = self.trainModel(X_train, Y_train_class, Y_train_dist, X_test, Y_test_class,Y_test_dist, unencoded_Y, test)
            cvscores = self.evaluator.handleScores(cvscores, scores, n_folds)
            foldNumber += 1

        print('Average results for the ten folds:')
        self.evaluator.prettyPrintResults(cvscores)
        ## write distance prediction stats to csv
        f = open('model_dist_predictions_ften.csv', 'w')
        f.write(csv_header + csv_entries_dist)
        f.close()

        if self.save_weights and model_type == 'baseline':
            self.model.save_weights('baseline_weights.h5')

        return cvscores

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
