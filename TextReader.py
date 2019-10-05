import os
import re
class TextReader:

    def __init__(self, baseDirectory):
        self.contents = []
        self.baseDirectory = baseDirectory

    def processText(self, fileName):
        file = open(fileName, "r", encoding='utf8')
        textContents = file.read()
        textContents = re.sub(r'\n', r' ', textContents)
        self.contents.append(textContents)

    def readTexts(self):
        fileList = os.listdir(self.baseDirectory)
        for file in fileList:
            fileName = self.baseDirectory + '/' + file
            self.processText(fileName)
