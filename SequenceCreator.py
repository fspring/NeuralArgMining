from keras.preprocessing.text import Tokenizer
class SequenceCreator:
    englishTextsMaxSize = 735

    def __init__(self, allTexts, textsToEval, useMaxlen):
        self.sequences = []
        self.maxlen = 0
        self.allTexts = allTexts
        self.textsToEval = textsToEval
        self.useMaxlen = useMaxlen

    def normalizeSequences(self):
        maxlen = 0
        newSequences = []
        for sequence in self.sequences:
            if len(sequence) > maxlen:
                maxlen = len(sequence)
        if self.useMaxlen:
            self.maxlen = maxlen
        else:
            self.maxlen = self.englishTextsMaxSize

        for sequence in self.sequences:
            while len(sequence) < self.maxlen:
                sequence.append(0)
            newSequences.append(sequence)
        self.sequences = newSequences

    def createSequences(self):
        token = Tokenizer(filters='')
        token.fit_on_texts(self.allTexts.contents)
        self.wordIndex = token.word_index
        self.sequences = token.texts_to_sequences(self.textsToEval.contents)
        self.normalizeSequences()

    def createTagSequences(self):
        token = Tokenizer(filters='')
        token.fit_on_texts(self.allTexts.contents)
        self.wordIndex = token.word_index
        print(self.wordIndex)
        self.sequences = token.texts_to_sequences(self.textsToEval.contents)
        self.normalizeSequences()

    def transformText(self, text, textList):
        token = Tokenizer(filters='')
        token.fit_on_texts(textList.contents)
        sequence = token.texts_to_sequences([text])
        sequences = []

        for text in sequence:
            while len(text) < self.maxlen:
                text.append(0)
            sequences.append(text)
            sequences.append(text)

        return sequences
