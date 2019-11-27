import TextReader as tr
import SequenceCreator as sc
import RelationTagProcessor as tp
import NeuralTrainer as nt

import numpy as np
import tensorflow as tf
import os
import datetime
import random
import argparse
import string
import keras

from keras import backend as K
from keras.layers import Layer

from utils import bool_flag

commonTextDirectory = 'allTextsPunctuation'
commonTagDirectory = 'allRelationTags'

englishTextsDirectory = 'essaysClaimsPremisesPunctuation/rel/texts'
englishTagDirectory = 'essaysClaimsPremisesPunctuation/rel/tags'

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

def preprocess_tags(textDirectory, tagDirectory, addTexts):
    ### PROCESS TEXT INPUT ###
    all_texts = tr.TextReader(commonTextDirectory)
    all_texts.readTexts()

    texts_to_eval = tr.TextReader(textDirectory)
    texts_to_eval.readTexts()

    textSequencer = sc.SequenceCreator(all_texts, texts_to_eval, not addTexts)

    textSequencer.create_sequences()
    text_sequences = textSequencer.sequences

    englishTextSequences = []
    if addTexts:
        englishTexts = tr.TextReader(englishTextsDirectory)
        englishTexts.readTexts()

        englishTextSequencer = sc.SequenceCreator(all_texts, englishTexts, False)

        englishTextSequencer.create_sequences()
        englishTextSequences = englishTextSequencer.sequences

    ### PROCESS TAG INPUT ###
    all_tags = tp.RelationTagProcessor(commonTagDirectory)
    all_tags.readTags()

    tags_to_eval = tp.RelationTagProcessor(tagDirectory)
    tags_to_eval.readTags()

    tagSequencer = sc.SequenceCreator(all_tags, tags_to_eval, not addTexts)

    tagSequencer.create_tag_sequences()
    unencoded_tags = tagSequencer.sequences

    tagSequencer.map_tag_encoding()

    tags_to_eval.num_tags = all_tags.num_tags = tagSequencer.n_tags

    tagSequencer.encode_onehot()
    tag_sequences = tagSequencer.encoded_sequences

    englishTagSequences = []
    if addTexts:
        english_tags = tp.TagProcessor(englishTagDirectory)
        english_tags.readTags()

        englishTagSequencer = sc.SequenceCreator(all_tags, english_tags, False)

        englishTagSequencer.create_tag_sequences()

        english_tags.num_tags = tagSequencer.n_tags

        englishTagSequencer.encode_onehot()
        englishTagSequences = englishTagSequencer.encoded_sequences

    return (textSequencer, tagSequencer, englishTextSequences, englishTagSequences, tags_to_eval)

def fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath):
    preprocess_start_time = datetime.datetime.now().replace(microsecond=0)

    (textSequencer, tagSequencer, englishTextSequences, englishTagSequences, tags_to_eval) = preprocess_tags(textDirectory, tagDirectory, addTexts)

    preprocess_end_time = datetime.datetime.now().replace(microsecond=0)
    print('Preprocessing time:', preprocess_end_time - preprocess_start_time)

    n_tags = tagSequencer.n_tags
    text_sequences = textSequencer.sequences
    tag_sequences = [tagSequencer.encoded_sequences, tags_to_eval.distance_tags_list]
    unencoded_tags = tagSequencer.sequences

    print(np.shape(text_sequences), np.shape(tag_sequences[0]), np.shape(unencoded_tags))
    print(textSequencer.maxlen)

    print(unencoded_tags[0])

    # print('Number of Tokens:', tags_to_eval.numNArg + tags_to_eval.numClaim + tags_to_eval.numPremise)
    # print('Number of Claims Tokens:', tags_to_eval.numClaim)
    # print('Number of Premises Tokens:', tags_to_eval.numPremise)

    # trainer = nt.NeuralTrainer(textSequencer.maxlen, n_tags, textSequencer.word_index, embeddings, textDirectory, dumpPath)
    # trainer.create_model()
    # # trainer.create_baseline_model()
    #
    # startTime = datetime.datetime.now().replace(microsecond=0)
    #
    # trainer.crossValidate(text_sequences, tag_sequences, englishTextSequences, englishTagSequences, unencoded_tags)
    #
    # endTime = datetime.datetime.now().replace(microsecond=0)
    # timeTaken = endTime - startTime
    #
    # print("Time elapsed:")
    # print(timeTaken)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate the argumentation mining models')
    parser.add_argument('model_name', type=str, default='', help='The abbreviature of the model to test: FtEn; FtPt; MSuEn; MUnEn; VMSuEn; VMUnEn; MSuPt; MUnPt; VMSuPt; VMUnPt; MSuEnPt; MUnEnPt; VMSuEnPt; VMUnEnPt')
    parser.add_argument('--cuda', type=bool_flag, default=False, help="Run on GPU")

    args = parser.parse_args()

    assert args.model_name in ["FtEn", "FtPt", "MSuEn", "MUnEn", "VMSuEn", "VMUnEn", "MSuPt", "MUnPt", "VMSuPt", "VMUnPt", "MSuEnPt", "MUnEnPt", "VMSuEnPt", "VMUnEnPt"]

    newDirectory = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    dumpPath = r'Dumps/' + newDirectory
    while os.path.exists(dumpPath):
        newDirectory = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        dumpPath = r'Dumps/' + newDirectory
    os.makedirs(dumpPath)
    os.makedirs(dumpPath + '_BCorr')
    # os.makedirs(dumpPath + '_baseline')

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(42)
    tf.set_random_seed(42)

    if 'EnPt' in args.model_name:
        textDirectory = 'CorpusOutputPunctuation/rel/texts'
        tagDirectory = 'CorpusOutputPunctuation/rel/tags'
        addTexts = True
    elif 'Pt' in args.model_name:
        textDirectory = 'CorpusOutputPunctuation/rel/texts'
        tagDirectory = 'CorpusOutputPunctuation/rel/tags'
        addTexts = False
    else:
        textDirectory = 'essaysClaimsPremisesPunctuation/rel/texts'
        tagDirectory = 'essaysClaimsPremisesPunctuation/rel/tags'
        addTexts = False

    embeddings = args.model_name + 'Emb'

    if args.cuda:
        with tf.device('/gpu:0'):
            fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath)
    else:
        with tf.device('/cpu:0'):
            fullSequence(textDirectory,tagDirectory, addTexts, embeddings, dumpPath)


if __name__ == '__main__':
    main()
