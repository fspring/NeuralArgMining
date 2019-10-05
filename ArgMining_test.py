import TextReader as tr
import SequenceCreator as sc
import TagProcessor as tp
import NeuralTrainer as nt

import numpy as np
import tensorflow as tf
import random as rn
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


def fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath, relations):
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

    if(relations):
        commonTagDirectory = 'basic_reltag'
    else:
        commonTagDirectory = 'allTagsPunctuation'

    allTags = tp.TagProcessor(commonTagDirectory, [0, 1, 0])
    allTags.readTags()

    tags = tp.TagProcessor(tagDirectory, [0, 1, 0])
    tags.readTags()
    if addTexts:
        tagSequencer = sc.SequenceCreator(allTags, tags, False)
    else:
        tagSequencer = sc.SequenceCreator(allTags, tags, True)
    tagSequencer.createTagSequences()
    unencodedTags = tagSequencer.sequences
    n_tags = max(tagSequencer.wordIndex.values())
    print(n_tags)
    tags.num_tags = n_tags
    nonarg_tag = np.zeros(n_tags)
    nonarg_tag[0] = 1
    tags.nArgTag = list(map(int, nonarg_tag))
    tagSequences = tags.encode(tagSequencer.sequences)

    model = nt.NeuralTrainer(textSequencer.maxlen, tags.num_tags, textSequencer.wordIndex,
                          embeddings, textDirectory, dumpPath, relations)
    startTime = datetime.datetime.now().replace(microsecond=0)

    if addTexts:
        englishTextsDirectory = 'essaysClaimsPremisesPunctuation/texts'
        englishTexts = tr.TextReader(englishTextsDirectory)
        englishTexts.readTexts()
        englishTextSequencer = sc.SequenceCreator(allTexts, englishTexts, False)
        englishTextSequencer.createSequences()
        englishTextSequences = englishTextSequencer.sequences

        if(relations):
            tagDirectory = 'essaysClaimsPremisesPunctuation/basic_reltag'
        else:
            tagDirectory = 'essaysClaimsPremisesPunctuation/tags'
        englishTags = tp.TagProcessor(englishTagDirectory, [0, 1, 0])
        englishTags.readTags()
        englishTagSequencer = sc.SequenceCreator(allTags, englishTags, False)
        englishTagSequencer.createTagSequences()
        englishTagSequences = englishTags.encode(englishTagSequencer.sequences)

        model.crossValidate(textSequences, tagSequences, englishTextSequences, englishTagSequences, unencodedTags)
    else:
        model.crossValidate(textSequences, tagSequences, [], [], unencodedTags)


    endTime = datetime.datetime.now().replace(microsecond=0)
    timeTaken = endTime - startTime

    print("Time elapsed:")
    print(timeTaken)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate the argumentation mining models')
    parser.add_argument('model_name', type=str, default='', help='The abbreviature of the model to test: FtEn; FtPt; MSuEn; MUnEn; VMSuEn; VMUnEn; MSuPt; MUnPt; VMSuPt; VMUnPt; MSuEnPt; MUnEnPt; VMSuEnPt; VMUnEnPt')
    parser.add_argument('--rel', type=bool_flag, default=False, help='Consider relations tags' )
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
    rn.seed(42)
    tf.set_random_seed(42)

    if 'EnPt' in args.model_name:
        textDirectory = 'CorpusOutputPunctuation/txt/texts'
        if(args.rel):
            tagDirectory = 'CorpusOutputPunctuation/txt/basic_reltag'
        else:
            tagDirectory = 'CorpusOutputPunctuation/txt/tags'
        addTexts = True
    elif 'Pt' in args.model_name:
        textDirectory = 'CorpusOutputPunctuation/txt/texts'
        if(args.rel):
            tagDirectory = 'CorpusOutputPunctuation/txt/basic_reltag'
        else:
            tagDirectory = 'CorpusOutputPunctuation/txt/tags'
        addTexts = False
    else:
        textDirectory = 'essaysClaimsPremisesPunctuation/texts'
        if(args.rel):
            tagDirectory = 'essaysClaimsPremisesPunctuation/basic_reltag'
        else:
            tagDirectory = 'essaysClaimsPremisesPunctuation/tags'
        addTexts = False

    embeddings = args.model_name + 'Emb'

    if args.cuda:
        with tf.device('/gpu:0'):
            fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath, args.rel)
    else:
        with tf.device('/cpu:0'):
            fullSequence(textDirectory,tagDirectory, addTexts, embeddings, dumpPath, args.rel)


if __name__ == '__main__':
    main()
