from keras.preprocessing.text import Tokenizer
import numpy as np
class SequenceCreator:

    def __init__(self, all_texts, texts_to_eval, use_maxlen):
        self.all_texts = all_texts
        self.texts_to_eval = texts_to_eval
        self.use_maxlen = use_maxlen

        self.english_text_maxsize = 735
        self.maxlen = 0

        self.sequences = []
        self.encoded_sequences = []

        self.word_index = None
        self.zero_tag = [0, 1, 0]
        self.n_tags = 0
        self.tag_code = {}

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

    def create_sequences(self):
        token = Tokenizer(filters='')
        token.fit_on_texts(self.all_texts.contents)
        self.word_index = token.word_index
        self.sequences = token.texts_to_sequences(self.texts_to_eval.contents)
        self.normalizeSequences()

    def create_tag_sequences(self):
        token = Tokenizer(filters='')
        token.fit_on_texts(self.all_texts.component_tags)
        self.word_index = token.word_index
        self.sequences = token.texts_to_sequences(self.texts_to_eval.component_tags)
        self.n_tags = max(token.word_index.values())
        self.normalizeSequences()
        self.normalize_distance_tag()

    def normalize_distance_tag(self):
        for i in range(0, len(self.texts_to_eval.distance_tags_list)):
            distance_tags = self.texts_to_eval.distance_tags_list[i]
            if '' in distance_tags:
                distance_tags.remove('')
            while len(distance_tags) < self.maxlen:
                distance_tags.append('0')
            for j in range(0, len(distance_tags)):
                if distance_tags[j] == '|':
                    distance_tags[j] = 0
                else:
                    value = int(distance_tags[j])
                    distance_tags[j] = value
            self.texts_to_eval.distance_tags_list[i] = distance_tags

    def map_tag_encoding(self):
        for i in range(0, len(self.texts_to_eval.component_tags)):
            full_tags = self.texts_to_eval.component_tags[i].split(' ')
            seq_tags = self.sequences[i]
            full_tags.remove('')
            for j in range(0, len(full_tags)):
                self.tag_code[full_tags[j]] = seq_tags[j]

        file = open('tag_mapping.txt', 'w', encoding='utf-8')
        for key in self.tag_code.keys():
            file.write(u'' + key + '\t' + str(self.tag_code[key]) + '\n')
        file.close()

    def encode_onehot(self):
        num_tags = max(self.word_index.values())
        for tag_seq in self.sequences:
            new_tags = []
            for tag in tag_seq:
                if tag == 0:
                    new_tags.append(self.zero_tag)
                else:
                    result = np.zeros(num_tags)
                    result[tag - 1] = 1
                    new_tags.append(list(map(int, result)))
            self.encoded_sequences.append(new_tags)
