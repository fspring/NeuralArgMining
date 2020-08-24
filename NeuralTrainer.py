import numpy as np
import os
import matplotlib.pyplot as plt
import sys

import Evaluator as ev
import RelationBaseline as rb
import PostProcessing as pp
import NeuralTrainerCustoms as ntc
import NeuralModel as nm

import AdaMod as am

from sklearn.model_selection import KFold

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils.generic_utils import get_custom_objects

from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

# np.set_printoptions(threshold=sys.maxsize)

class NeuralTrainer:
    embedding_size = 300
    hidden_size = 100


    def __init__(self, maxlen, num_tags, word_index, embeddings, model_type, texts_to_eval_dir, dumpPath):
        self.num_tags = num_tags
        self.word_index = word_index
        self.texts_to_eval_dir = texts_to_eval_dir
        self.dumpPath = dumpPath
        self.model_maker = nm.NeuralModel(maxlen, num_tags, word_index, embeddings)
        num_measures = 1 + 3*(num_tags - 2)
        self.evaluator = ev.Evaluator(num_tags, num_measures, self.model_maker.tags)
        self.postprocessing = pp.PostProcessing(num_tags, self.model_maker.tags)

    def make_model(self, maxlen, num_tags, word_index, embeddings, model_type, fold_name):
        # print('model_type:', model_type) #debug
        if model_type == 'baseline':
            self.model_maker.create_baseline_model()
        elif model_type == 'crf_dist':
            self.model_maker.create_model()
        elif model_type == 'dual':
            self.model_maker.save_weights = True
            self.model_maker.create_model(fold_name=fold_name)

    def train_baseline_model(self, x_train, y_train, x_test, y_test_class, y_test_dist, unencodedY, testSet, fold_name):
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='min')
        checkpoint_filepath = './tmp/'+fold_name+'/baseline_checkpoint.h5'
        if not os.path.exists('tmp/'+fold_name):
            os.makedirs('tmp/'+fold_name)

        mcp_save = ModelCheckpoint(checkpoint_filepath, save_weights_only=True, save_best_only=True, monitor='loss', mode='min', verbose=1)

        print('x train shape:', x_train.shape, 'x test shape:', x_test.shape)
        print('y train shape:', y_train.shape, 'y test shape:', y_test_class.shape)

        self.model_maker.model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor, mcp_save])
        self.model_maker.model.load_weights(checkpoint_filepath)

        scores = self.model_maker.model.evaluate(x_test, y_test_class, batch_size=8, verbose=1)
        y_pred_class = self.model_maker.model.predict(x_test)

        # scores: loss, crf_acc
        print("%s: %.2f%%" % (self.model_maker.model.metrics_names[1], scores[1] * 100))

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

        (spanEvalAt100, correct_spans_at_100) = self.evaluator.spanEval(pred_spans, true_spans, 1.0)
        (spanEvalAt075, correct_spans_at_075) = self.evaluator.spanEval(pred_spans, true_spans, 0.75)
        (spanEvalAt050, correct_spans_at_050) = self.evaluator.spanEval(pred_spans, true_spans, 0.50)

        print('------- Distances Baseline -------')
        dist_eval_at_100 = self.evaluator.dist_eval(pred_spans, true_spans, b_pred_dist, y_test_dist, correct_spans_at_100, 1.0)
        dist_eval_at_075 = self.evaluator.dist_eval(pred_spans, true_spans, b_pred_dist, y_test_dist, correct_spans_at_075, 0.75)
        dist_eval_at_050 = self.evaluator.dist_eval(pred_spans, true_spans, b_pred_dist, y_test_dist, correct_spans_at_050, 0.50)
        print('----------------------------------')

        print('------- Clusters Baseline -------')
        b_cubed_eval_at_100 = self.evaluator.b_cubed_span_eval(pred_spans, true_spans, b_pred_dist, y_test_dist, correct_spans_at_100, 1.0)
        b_cubed_eval_at_075 = self.evaluator.b_cubed_span_eval(pred_spans, true_spans, b_pred_dist, y_test_dist, correct_spans_at_075, 0.75)
        b_cubed_eval_at_050 = self.evaluator.b_cubed_span_eval(pred_spans, true_spans, b_pred_dist, y_test_dist, correct_spans_at_050, 0.50)
        print('---------------------------------')

        tagEval = self.evaluator.tagEval(pred_spans, true_spans)

        return [scores[1], tagEval, spanEvalAt100, spanEvalAt075, spanEvalAt050,
                dist_eval_at_100, dist_eval_at_075, dist_eval_at_050,
                b_cubed_eval_at_100, b_cubed_eval_at_075, b_cubed_eval_at_050]

    def trainModel(self, x_train, y_train_class, y_train_dist, y_target, x_test, y_test_class, y_test_dist, unencodedY, testSet, fold_name):
        f1_metric = ntc.F1_Metric(y_target, x_train, y_train_dist, self.postprocessing, self.evaluator, verbose=1)

        monitor_measure = 'loss'
        monitor_mode = 'min'

        monitor = ntc.CustomEarlyStopping(monitor=monitor_measure, min_delta=0.001, patience=5, verbose=1, mode=monitor_mode, threshold=None)
        # monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='min')

        checkpoint_filepath = '/tmp/model_checkpoint.h5'
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        mcp_save = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True, monitor=monitor_measure, mode=monitor_mode)

        y_train = [y_train_class,y_train_dist]
        epochs = 100
        history = self.model_maker.model.fit(x_train, y_train, epochs=epochs, batch_size=8, verbose=1, callbacks=[f1_metric, monitor, mcp_save])
        history_plus = None
        swicth_func = 0
        epochs_run = len(history.history['loss'])
        if epochs_run < 100:
            swicth_func = epochs_run - 1
            self.model_maker.recompile_model_new_loss('consecutive_dist_loss', fold_name)

            epochs_left = epochs - epochs_run
            history_plus = self.model_maker.model.fit(x_train, y_train, epochs=epochs_left, batch_size=8, verbose=1, callbacks=[f1_metric, monitor, mcp_save])

        self.model_maker.model.load_weights(checkpoint_filepath)
        if history_plus:
            # print(history_plus.history.keys())
            additional_loss = history_plus.history['loss']
            additional_softargmax_loss = history_plus.history['softargmax_loss']
            additional_crf_loss = history_plus.history['crf_layer_loss']
            additional_softargmax_f1 = history_plus.history['f1']
        else:
            additional_loss = []
            additional_softargmax_loss = []
            additional_crf_loss = []
            additional_softargmax_f1 = []

        plt.figure()
        plt.suptitle('Loss switched at epoch ' + str(swicth_func))
        plt.subplot(411)
        plt.title('Loss')
        plt.plot(history.history['loss']+additional_loss, label='train')
        plt.legend()

        plt.subplot(412)
        plt.title('Dist Loss')
        plt.plot(history.history['softargmax_loss']+additional_softargmax_loss, label='train')
        plt.legend()

        plt.subplot(413)
        plt.title('CRF Loss')
        plt.plot(history.history['crf_layer_loss']+additional_crf_loss, label='train')
        plt.legend()

        plt.subplot(414)
        plt.title('Dist F1')
        plt.plot(history.history['f1']+additional_softargmax_f1, label='train')
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.savefig(fold_name + '_train_eval.png')


        layer = self.model_maker.model.get_layer('crf_layer')
        weights = layer.get_weights()
        gen_matrix = weights[1]

        # print('before:', gen_matrix) #debug

        for i in range(0, self.num_tags):
            for j in range(0, self.num_tags):
                if self.model_maker.transition_matrix[i][j] == 1:
                    gen_matrix[i][j] = 1

        # print('after:', gen_matrix) #debug
        weights[1] = gen_matrix

        self.model_maker.model.get_layer('crf_layer').set_weights(weights)
        # print('check:', self.model_maker.model.get_layer('crf_layer').get_weights()[1]) #debug

        y_test = [y_test_class,y_test_dist]
        scores = self.model_maker.model.evaluate(x_test, y_test, batch_size=8, verbose=1)
        [y_pred_class, y_pred_dist] = self.model_maker.model.predict(x_test)

        # scores: loss, crf_loss, dist_loss, crf_acc, dist_acc
        # print("%s: %.2f%%" % (self.model_maker.model.metrics_names[3], scores[3] * 100))
        # print("%s: %.2f%%" % (self.model_maker.model.metrics_names[4], scores[4] * 100))
        # print("%s: %.2f%%" % (self.model_maker.model.metrics_names[2], scores[2] * 100))
        if not os.path.exists(self.dumpPath + '_BCorr'):
            os.makedirs(self.dumpPath + '_BCorr')
        self.write_evaluated_tests_to_file(x_test, y_pred_class, y_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath + '_BCorr')

        (y_pred_class, c_pred_dist) = self.postprocessing.correct_dist_prediction(y_pred_class, y_pred_dist, unencodedY)

        self.write_evaluated_tests_to_file(x_test, y_pred_class, c_pred_dist, testSet, self.texts_to_eval_dir, self.dumpPath)

        (true_spans, pred_spans) = self.postprocessing.replace_argument_tag(y_pred_class, unencodedY)

        (spanEvalAt100, correct_spans_at_100) = self.evaluator.spanEval(pred_spans, true_spans, 1.0)
        (spanEvalAt075, correct_spans_at_075) = self.evaluator.spanEval(pred_spans, true_spans, 0.75)
        (spanEvalAt050, correct_spans_at_050) = self.evaluator.spanEval(pred_spans, true_spans, 0.50)

        print('------- Distances from model -------')
        dist_eval_at_100 = self.evaluator.dist_eval(pred_spans, true_spans, c_pred_dist, y_test_dist, correct_spans_at_100, 1.0)
        dist_eval_at_075 = self.evaluator.dist_eval(pred_spans, true_spans, c_pred_dist, y_test_dist, correct_spans_at_075, 0.75)
        dist_eval_at_050 = self.evaluator.dist_eval(pred_spans, true_spans, c_pred_dist, y_test_dist, correct_spans_at_050, 0.5)
        print('------------------------------------')

        print('------- Clusters Baseline -------')
        b_cubed_eval_at_100 = self.evaluator.b_cubed_span_eval(pred_spans, true_spans, c_pred_dist, y_test_dist, correct_spans_at_100, 1.0)
        b_cubed_eval_at_075 = self.evaluator.b_cubed_span_eval(pred_spans, true_spans, c_pred_dist, y_test_dist, correct_spans_at_075, 0.75)
        b_cubed_eval_at_050 = self.evaluator.b_cubed_span_eval(pred_spans, true_spans, c_pred_dist, y_test_dist, correct_spans_at_050, 0.50)
        print('---------------------------------')

        tagEval = self.evaluator.tagEval(pred_spans, true_spans)

        return [scores[3], tagEval, spanEvalAt100, spanEvalAt075, spanEvalAt050,
                dist_eval_at_100, dist_eval_at_075, dist_eval_at_050,
                b_cubed_eval_at_100, b_cubed_eval_at_075, b_cubed_eval_at_050]

    def crossValidate(self, X, Y, additionalX, additionalY, unencodedY, additional_unencodedY, model_type):
        seed = 42
        n_folds = 10
        foldNumber  = 1


        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

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
            Y_target = []
            for trainIndex in train:
                X_train.append(X[trainIndex])
                Y_train_class.append(Y_class[trainIndex])
                Y_train_dist.append(Y_dist[trainIndex])
                Y_target.append(unencodedY[trainIndex])

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
            Y_target = Y_target + additional_unencodedY

            X_train = np.array(X_train)
            Y_train_class = np.array(Y_train_class)
            Y_train_dist = np.array(Y_train_dist)
            X_test = np.array(X_test)
            Y_test_class = np.array(Y_test_class)
            Y_test_dist = np.array(Y_test_dist)
            unencoded_Y = np.array(unencoded_Y)

            # print('after additional x shape:', X_train.shape)
            # print('after additional y class shape:', Y_train_class.shape)
            # print('after additional y dist shape:', Y_train_dist.shape)
            # print('test x shape:', X_test.shape)
            # print('test y class index:', Y_test_class.shape)
            # print('test y dist index:', Y_test_dist.shape)

            self.make_model(self.model_maker.maxlen, self.num_tags, self.word_index, self.model_maker.embeddings, model_type, 'fold_'+str(foldNumber))

            scores = None
            if model_type == 'baseline':
                scores = self.train_baseline_model(X_train, Y_train_class, X_test, Y_test_class,Y_test_dist, unencoded_Y, test, 'fold_'+str(foldNumber))
            else:
                scores = self.trainModel(X_train, Y_train_class, Y_train_dist, Y_target, X_test, Y_test_class,Y_test_dist, unencoded_Y, test, 'fold_'+str(foldNumber))
            cvscores = self.evaluator.handleScores(cvscores, scores, n_folds)
            foldNumber += 1

        print('Average results for the ten folds:')
        self.evaluator.prettyPrintResults(cvscores)

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
                textFile.write(u'' + token[0] + ' ' + self.model_maker.tags[token[1]] + ' ' + '{:8.3f}'.format(token[2][0]) + '\n')
