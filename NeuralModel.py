import numpy as np
import os
# import pathlib
import matplotlib.pyplot as plt
import sys

import NeuralTrainerCustoms as ntc

import AdaMod as am

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Layer, Dense, Activation, Embedding, LSTM, Bidirectional, Lambda, concatenate
from keras.layers.wrappers import TimeDistributed

import keras.losses

# import tensorflow as tf

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from keras.utils.generic_utils import get_custom_objects

# np.set_printoptions(threshold=sys.maxsize)

# losses = ntc.Losses()
# crf_tensor_arg = K.zeros((129, 741, 4))
# get_custom_objects().update({'consecutive_dist_loss': losses.consecutive_dist_loss_wrapper(crf_tensor_arg)})
# get_custom_objects().update({'loss_func': losses.loss_func})
# keras.losses.loss_func = losses.loss_func

class NeuralModel:
    embedding_size = 300
    hidden_size = 100


    def __init__(self, maxlen, num_tags, word_index, embeddings, save_weights=False):
        self.maxlen = maxlen
        self.vocab_size = len(word_index)+1
        self.num_tags = num_tags
        self.word_index = word_index
        self.embeddings = embeddings
        self.model = None
        self.tags = ['']*num_tags
        self.arg_classes = ['']*num_tags
        self.transition_matrix = None
        self.save_weights = save_weights
        self.read_tag_mapping()
        self.set_transition_matrix()
        num_measures = 1 + 3*(num_tags - 2)

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

    def switch_loss_wrapper(self, crf_layer):
        self.current_epoch = K.variable(100)
        current_epoch = self.current_epoch

        def switch_loss(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)

            pure_mae = K.mean(K.abs(y_pred - y_true), axis=-1)
            y_true_aux = K.squeeze(y_true, axis=-1)

            zero = K.constant(0)

            simple_loss = K.switch(K.equal(y_true_aux, zero), K.zeros_like(pure_mae), pure_mae)

            # print('ypred shape', K.int_shape(y_pred))

            I_prob = K.squeeze(crf_layer[:,:,:1], axis=-1)

            ypred_size = K.int_shape(y_pred)[1]
            tiled = K.tile(y_pred, [1, 2, 1]) #repeat array like [1, 2, 3] -> [1, 2, 3, 1, 2, 3]
            rolled_y_pred = tiled[:,ypred_size-1:-1] #crop repeated array (from len-1) -> [3, 1, 2] <- (to -1)

            dist_dif = K.abs((rolled_y_pred - y_pred) - K.ones_like(y_pred))

            dist_err_mae = K.switch(K.greater(I_prob, K.constant(0.5)), K.mean(K.abs(y_pred - y_true + dist_dif), axis=-1), K.mean(K.abs(y_pred - y_true), axis=-1))


            dist_err_loss = K.switch(K.equal(y_true_aux, zero), K.zeros_like(dist_err_mae), dist_err_mae)

            K.switch(K.less(current_epoch, K.constant(100)), dist_err_loss, simple_loss)

        return switch_loss

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

        soft_argmax = ntc.SoftArgMax()
        soft_argmax.create_soft_argmax_layer()

        # zero_switch = ntc.SoftArgMax()
        # zero_switch.create_zero_switch_layer()

        concat = concatenate([crf_tensor, dist_tensor], axis=-1, name='concatenate')

        ### LAYER OPTIONS:
        ##### soft_argmax.layer
        ##### zero_switch.layer
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
        ####LOSSES:
        ######'mean_absolute_error'
        ######'loss_func'
        ######'consecutive_dist_loss'
        ####OPTIMIZERS:
        ######'adam'
        ######am.AdaMod() ??
        # get_custom_objects().update({'consecutive_dist_loss': losses.consecutive_dist_loss_wrapper(crf_tensor)})
        # get_custom_objects().update({'switch_loss': losses.switch_loss_wrapper(crf_tensor)})
        #
        # keras.losses.consecutive_dist_loss = losses.consecutive_dist_loss_wrapper(crf_tensor)
        self.model.compile(optimizer='adam', loss=[crf_loss,self.switch_loss_wrapper(crf_tensor)], loss_weights=[1.0, 0.10], metrics={'crf_layer':[crf_accuracy], 'softargmax':'mae'})

        # self.model.run_eagerly = True #debug

        if self.save_weights:
            print('MODEL LOADED FROM FILE')

            base_crf_tensor = self.create_CRF(biLSTM_tensor, 'join', 'viterbi')
            baseline_model = Model(input=input, output=base_crf_tensor)
            print(baseline_model.summary()) #debug
            baseline_model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_accuracy])

            # baseline_model.run_eagerly = True #debug

            baseline_model.load_weights('baseline_weights.h5', by_name=True)

            base_layers = baseline_model.layers
            model_layers = self.model.layers
            for i in range(0, len(base_layers)):
                print(model_layers[i].name, base_layers[i].name)
                assert model_layers[i].name == base_layers[i].name
                layer_name = base_layers[i].name
                self.model.get_layer(layer_name).set_weights(base_layers[i].get_weights())
                self.model.get_layer(layer_name).trainable = False

    def create_baseline_model(self):
        input = Input(shape=(self.maxlen,))

        biLSTM_tensor = self.create_biLSTM(input)
        crf_tensor = self.create_CRF(biLSTM_tensor, 'join', 'viterbi')

        self.model = Model(input=input, output=crf_tensor)
        print(self.model.summary()) #debug

        self.model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_accuracy])

        # self.model.run_eagerly = True #debug
