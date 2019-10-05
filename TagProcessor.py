import os
import re
import numpy as np
class TagProcessor:
    def __init__(self, baseDirectory, nArgTag):
        self.contents = []
        self.num_tags = 3
        self.baseDirectory = baseDirectory
        self.nArgTag = nArgTag

        self.numNArg = 0
        self.numClaim = 0
        self.numPremise = 0

    def encode(self, tagList):
        encodedTags = []
        for tags in tagList:
            newTags = []
            for tag in tags:
                if tag == 0:
                    newTags.append(self.nArgTag)
                else:
                    result = np.zeros(self.num_tags)
                    result[tag - 1] = 1
                    newTags.append(list(map(int, result)))
            encodedTags.append(newTags)
        return encodedTags

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
            elif tag[1] == 'O':
                self.numNArg += 1
            elif tag[3:-1] == 'premise':
                self.numPremise += 1
            elif tag[3:-1] == 'claim':
                self.numClaim += 1
