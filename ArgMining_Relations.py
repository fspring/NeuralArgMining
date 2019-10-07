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


def fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath):
    commonTextDirectory = 'allTextsPunctuation'
    allTexts = tr.TextReader(commonTextDirectory)
    allTexts.readTexts()

    texts = tr.TextReader(textDirectory)
    texts.readTexts()
    if addTexts:
        textSequencer = sc.SequenceCreator(allTexts, texts, False)
    else:
        textSequencer = sc.SequenceCreator(allTexts, texts, True)
    textSequencer.createSequences()
    textSequences = textSequencer.sequences

    commonTagDirectory = 'allRelationTags'

    allTags = tp.RelationTagProcessor(commonTagDirectory)
    allTags.readTags()

    tags = tp.RelationTagProcessor(tagDirectory)
    tags.readTags()

    if addTexts:
        tagSequencer = sc.SequenceCreator(allTags, tags, False)
    else:
        tagSequencer = sc.SequenceCreator(allTags, tags, True)

    tagSequencer.createSequences()
    unencodedTags = tagSequencer.sequences
    n_tags = max(tagSequencer.word_index.values())
    tags.num_tags = allTags.num_tags = n_tags

    nonarg_tag = np.zeros(n_tags)
    nonarg_tag[0] = 1
    tags.nArgTag = list(map(int, nonarg_tag))
    allTags.nArgTag = list(map(int, nonarg_tag))

    allTags.map_encoding(tagSequencer.sequences)
    tagSequences = allTags.encode(tagSequencer.sequences)

    trainer = nt.NeuralTrainer(textSequencer.maxlen, n_tags, textSequencer.wordIndex,
                          embeddings, textDirectory, dumpPath)
    trainer.create_biLSTM_CRF_model()
    startTime = datetime.datetime.now().replace(microsecond=0)
    
    if addTexts:
        englishTextsDirectory = 'essaysClaimsPremisesPunctuation/rel/texts'
        englishTexts = tr.TextReader(englishTextsDirectory)
        englishTexts.readTexts()
        englishTextSequencer = sc.SequenceCreator(allTexts, englishTexts, False)
        englishTextSequencer.createSequences()
        englishTextSequences = englishTextSequencer.sequences

        tagDirectory = 'essaysClaimsPremisesPunctuation/rel/tags'
        englishTags = tp.TagProcessor(englishTagDirectory, [0, 1, 0])
        englishTags.readTags()
        englishTagSequencer = sc.SequenceCreator(allTags, englishTags, False)
        englishTagSequencer.createTagSequences()
        englishTagSequences = englishTags.encode(englishTagSequencer.sequences)

        trainer.crossValidate(textSequences, tagSequences, englishTextSequences, englishTagSequences, unencodedTags)
    else:
        trainer.crossValidate(textSequences, tagSequences, [], [], unencodedTags)


    endTime = datetime.datetime.now().replace(microsecond=0)
    timeTaken = endTime - startTime

    print("Time elapsed:")
    print(timeTaken)


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
