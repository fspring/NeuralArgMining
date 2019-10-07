from keras.preprocessing.text import Tokenizer
class SequenceCreator:
    english_text_maxsize = 735
    sequences = []
    maxlen = 0
    word_index = None

    def __init__(self, all_texts, texts_to_eval, use_maxlen):
        self.all_texts = all_texts
        self.texts_to_eval = texts_to_eval
        self.use_maxlen = use_maxlen

    def normalizeSequences(self):
        maxlen = 0
        new_sequences = []
        for sequence in self.sequences:
            if len(sequence) > maxlen:
                maxlen = len(sequence)
        if self.use_maxlen:
            self.maxlen = maxlen
        else:
            self.maxlen = self.english_text_maxsize

        for sequence in self.sequences:
            while len(sequence) < self.maxlen:
                sequence.append(0)
            new_sequences.append(sequence)
        self.sequences = new_sequences

    def createSequences(self):
        token = Tokenizer(filters='')
        token.fit_on_texts(self.all_texts.contents)
        self.word_index = token.word_index
        self.sequences = token.texts_to_sequences(self.texts_to_eval.contents)
        self.normalizeSequences()
