import os
import re
import numpy as np
class RelationTagProcessor:
    contents = []
    num_tags = 0
    nArgTag = None

    numNArg = 0
    numClaim = 0
    numPremise = 0

    tag_code = {}

    def __init__(self, baseDirectory):
        self.baseDirectory = baseDirectory

    def processTags(self, fileName):
        file = open(fileName, "r", encoding='utf8')
        TagContents = file.read()
        TagContents = re.sub(r'\n', r' ', TagContents)
        self.contents.append(TagContents)
        self.tag_count(TagContents)

    def readTags(self):
        fileList = os.listdir(self.baseDirectory)
        for file in fileList:
            fileName = self.baseDirectory + '/' + file
            self.processTags(fileName)

    def tag_count(self, tags_list):
        tags = tags_list.split(' ')
        for tag in tags:
            if tag == '':
                continue
            tag_elements = tag.split(',')
            if tag_elements[0][1] == 'O':
                self.numNArg += 1
            elif tag_elements[1] == 'premise':
                self.numPremise += 1
            elif tag_elements[1] == 'claim':
                self.numClaim += 1

    def map_encoding(self, tag_sequences):
        for i in range(0, len(self.contents)):
            full_tags = self.contents[i].split(' ')
            seq_tags = tag_sequences[i]
            if len(full_tags) != len(seq_tags):
                full_tags.remove('')
            for j in range(0, len(full_tags)):
                self.tag_code[full_tags[j]] = seq_tags[j]
        file = open('tag_mapping.txt', 'w', encoding='utf-8')
        for key in self.tag_code.keys():
            file.write(u'' + key + '\t' + str(self.tag_code[key]) + '\n')
        file.close()

    def encode(self, tag_sequences):
        encodedTags = []
        for tag_seq in tag_sequences:
            new_tags = []
            for tag in tag_seq:
                if tag == 0:
                    new_tags.append(self.nArgTag)
                else:
                    result = np.zeros(self.num_tags)
                    result[tag - 1] = 1
                    new_tags.append(list(map(int, result)))
            encodedTags.append(new_tags)
        return encodedTags
