import numpy as np
import keras
from keras import backend as K
from keras.layers import Lambda, Layer
from keras.activations import softmax
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

LOSS_UPDATE_CONTROL = 100

class Pentanh(Layer):

    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'

    def call(self, inputs):
        return K.switch(K.greater(inputs,0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self):
        return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape

keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})

class SoftArgMax:
    def __init__(self):
        self.layer = None

    def soft_argmax_func(self, x, beta=1e10):
        x_class = x[:,:4]
        x_dist = x[:,4:]

        x_range = K.arange(0,x_class.shape.as_list()[-1], dtype=x_class.dtype)
        output = K.sum(softmax(x_class*beta) * x_range, axis=-1, keepdims=False)
        # print('output shape', K.int_shape(output))

        # output = tf.reduce_sum(softmax(x_class*beta) * x_range, axis=-1)
        # print('output shape', K.int_shape(output))

        lower = K.constant(0.5)
        upper = K.constant(1.5)
        zero = K.zeros_like(x_dist)
        #if 0.5 <= output < 1.5 then return 0 else return predicted distance
        return K.switch(K.all(K.stack([K.greater_equal(output, lower), K.less(output, upper)], axis=0), axis=0), zero, x_dist)

    def create_soft_argmax_layer(self):
        self.layer = Lambda(self.soft_argmax_func, output_shape=(1,), name='lambda_softargmax')

    def zero_switch_func(self, x, beta=1e10):
        # x = tf.split(x, [4, 1], -1)
        x_class = x[:,:4]
        x_dist = x[:,4:]

        O_prob = x_class[:,1]
        zero = K.zeros_like(x_dist)

        #if 0.5 <= output < 1.5 then return 0 else return predicted distance
        return K.switch(K.less(O_prob, K.constant(0.25)), x_dist, zero)

    def create_zero_switch_layer(self):
        self.layer = Lambda(self.zero_switch_func, output_shape=(1,), name='lambda_zero_switch')

class Losses:
    def consecutive_dist_loss_wrapper(self, crf_layer):
        def consecutive_dist_loss(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)

            # print('ypred shape', K.int_shape(y_pred))

            I_prob = K.squeeze(crf_layer[:,:,:1], axis=-1)

            ypred_size = K.int_shape(y_pred)[1]
            tiled = K.tile(y_pred, [1, 2, 1]) #repeat array like [1, 2, 3] -> [1, 2, 3, 1, 2, 3]
            rolled_y_pred = tiled[:,ypred_size-1:-1] #crop repeated array (from len-1) -> [3, 1, 2] <- (to -1)

            dist_dif = K.abs((rolled_y_pred - y_pred) - K.ones_like(y_pred))

            mae = K.switch(K.greater(I_prob, K.constant(0.5)), K.mean(K.abs(y_pred - y_true + dist_dif), axis=-1), K.mean(K.abs(y_pred - y_true), axis=-1))

            y_true_aux = K.squeeze(y_true, axis=-1)
            zero = K.constant(0)

            return K.switch(K.equal(y_true_aux, zero), K.zeros_like(mae), mae)
        return consecutive_dist_loss


    def loss_func(self, y_true, y_pred):
        if not K.is_tensor(y_pred):
            y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)

        mae = K.mean(K.abs(y_pred - y_true), axis=-1)

        y_true = K.squeeze(y_true, axis=-1)
        zero = K.constant(0)

        return K.switch(K.equal(y_true, zero), K.zeros_like(mae), mae)

    def switch_loss_wrapper(self, crf_layer):
        def consecutive_dist_loss(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)

            # print('ypred shape', K.int_shape(y_pred))

            I_prob = K.squeeze(crf_layer[:,:,:1], axis=-1)

            ypred_size = K.int_shape(y_pred)[1]
            tiled = K.tile(y_pred, [1, 2, 1]) #repeat array like [1, 2, 3] -> [1, 2, 3, 1, 2, 3]
            rolled_y_pred = tiled[:,ypred_size-1:-1] #crop repeated array (from len-1) -> [3, 1, 2] <- (to -1)

            dist_dif = K.abs((rolled_y_pred - y_pred) - K.ones_like(y_pred))

            mae = K.switch(K.greater(I_prob, K.constant(0.5)), K.mean(K.abs(y_pred - y_true + dist_dif), axis=-1), K.mean(K.abs(y_pred - y_true), axis=-1))

            y_true_aux = K.squeeze(y_true, axis=-1)
            zero = K.constant(0)

            return K.switch(K.equal(y_true_aux, zero), K.zeros_like(mae), mae)

        def loss_func(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
            y_true = K.cast(y_true, y_pred.dtype)

            mae = K.mean(K.abs(y_pred - y_true), axis=-1)

            y_true = K.squeeze(y_true, axis=-1)
            zero = K.constant(0)

            return K.switch(K.equal(y_true, zero), K.zeros_like(mae), mae)

        if LOSS_UPDATE_CONTROL < 100:
            return consecutive_dist_loss
        else:
            return loss_func



class F1_Metric(keras.callbacks.Callback):
    def __init__(self, y_target, x_train, y_train_dist, postprocessing, evaluator, verbose=0):
        self.verbose = verbose
        self.y_target = y_target
        self.x_train = x_train
        self.y_train_dist = y_train_dist
        self.postprocessing = postprocessing
        self.evaluator = evaluator

    def on_epoch_begin(self, epoch, logs=None):
        self.f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        [y_pred_class, y_pred_dist] = self.model.predict(self.x_train)

        (y_pred_class, c_pred_dist) = self.postprocessing.correct_dist_prediction(y_pred_class, y_pred_dist, self.y_target)

        (true_spans, pred_spans) = self.postprocessing.replace_argument_tag(y_pred_class, self.y_target)

        (spanEvalAt100, correct_spans_at_100) = self.evaluator.spanEval(pred_spans, true_spans, 1.0, verbose=False)

        dist_eval_at_100 = self.evaluator.dist_eval(pred_spans, true_spans, c_pred_dist, self.y_train_dist, correct_spans_at_100, 1.0, verbose=False)

        self.f1 += dist_eval_at_100[2]

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["f1"] = dist_eval_at_100[2]
        if self.verbose:
            print('f1', self.f1)

        return

class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor='loss', min_delta=0.001, patience=1, mode='min', verbose=0, threshold=None):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.current_epoch = K.variable(100)
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        if self.monitor_op == np.less:
            self.best = np.Inf
        else:
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Loss Function Update requires %s available!" % self.monitor, RuntimeWarning)

        if self.threshold == None or self.monitor_op(current, self.threshold):
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stop' % (self.stopped_epoch + 1))
