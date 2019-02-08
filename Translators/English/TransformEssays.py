import re
import os
import datetime

class Word:
    def __init__(self, content, tag):
        self.content = content
        self.tag = tag

class Component:
    def setDistance(self, distance):
        self.distance = distance
        self.startPosition = -1
        self.position = -1

class Premise(Component):
    def __init__(self, words):
        self.words = words

    def getText(self):
        words = []
        for word in self.words:
            words.append(word.content)
        return ' '.join(words)

    def getTags(self):
        words = []
        for word in self.words:
            words.append(word.tag)
        return ' '.join(words)


class Claim(Component):
    def __init__(self, words):
        self.words = words

    def getText(self):
        words = []
        for word in self.words:
            words.append(word.content)
        return ' '.join(words)

    def getTags(self):
        words = []
        for word in self.words:
            words.append(word.tag)
        return ' '.join(words)

class ReadComponents:
    def __init__(self, fileName):
        self.claims = []
        self.premises = []
        self.fileName = "brat-project-final/" + fileName

    def readComponents(self):
        with open(self.fileName, "r", encoding='utf8') as file:
            for line in file:
                line = re.sub(r'[\n]+(?![0-9])', r'', line)
                line = re.sub(r'[.]+(?![0-9])', r' .', line)
                line = re.sub(r'[:]+(?![0-9])', r' :', line)
                line = re.sub(r'[,]+(?![0-9])', r' ,', line)
                line = re.sub(r'[;]+(?![0-9])', r' ;', line)
                line = re.sub(r'[?]+(?![0-9])', r' ?', line)
                line = re.sub(r'[!]+(?![0-9])', r' !', line)
                line = self.stripBlockTags(line)
                if(self.deleteLine(line) != True):
                    self.normalizeStructures(line)

    def stripBlockTags(self, line):
        return line.split("\t", 1)[1]

    def deleteLine(self, line):
        component = line.split(" ", 1)[0]
        if(component == 'MajorClaim'):
            return False
        elif(component == 'Premise'):
            return False
        elif(component == 'Claim'):
            return False
        else:
            return True

    def normalizeStructures(self, line):
        componentType = line.split(" ", 1)[0]
        sentence = line.split("\t", 1)[1]

        if ((componentType == 'MajorClaim') or (componentType == 'Claim')):
            tag = '(I,claim)'
        else:
            tag = '(I,premise)'

        taggedWords = []
        for word in sentence.split(" "):
            taggedWords.append(Word(word,tag))
        if ((componentType == 'MajorClaim') or (componentType == 'Claim')):
            self.claims.append(Claim(taggedWords))
        else:
            self.premises.append(Premise(taggedWords))

class ReadText():
    def __init__(self, fileName):
        self.words =  []
        self.fileName = "brat-project-final/" + fileName

    def readText(self):
        lineNumber = 1
        with open(self.fileName, "r", encoding='utf8') as file:
            for line in file:
                line = re.sub(r'[\n]+(?![0-9])', r'', line)
                line = re.sub(r'[.]+(?![0-9])', r' .', line)
                line = re.sub(r'[:]+(?![0-9])', r' :', line)
                line = re.sub(r'[,]+(?![0-9])', r' ,', line)
                line = re.sub(r'[;]+(?![0-9])', r' ;', line)
                line = re.sub(r'[?]+(?![0-9])', r' ?', line)
                line = re.sub(r'[!]+(?![0-9])', r' !', line)
                if ((lineNumber == 1) or (lineNumber == 2)):
                    lineNumber = lineNumber + 1
                    continue
                else:
                    for word in line.split(" "):
                        if(len(word) > 0):
                            self.words.append(Word(word,'(O,|)'))


class ComponentsReplacer():
    def __init__(self, originalText, existingClaimsAndPremises):
        self.originalText = originalText
        self.existingClaimsAndPremises = existingClaimsAndPremises
        self.processedText = originalText

    def getText(self):
        words = []
        for word in self.originalText:
            words.append(word.content)
        return ' '.join(words)

    def getProcessedText(self):
        words = []
        for word in self.processedText:
            words.append(word.content)
        return ' '.join(words)

    def getTags(self):
        tags = []
        for word in self.processedText:
            tags.append(word.tag)
        return ' '.join(tags)


    def matchText(self, wordPosition, component):
        isMatch = True

        for word in component.words:
            if (wordPosition +1 > len(self.originalText)):
                isMatch = False
                break
            elif (word.content != self.originalText[wordPosition].content):
                isMatch = False
                break
            wordPosition += 1

        return isMatch

    def replaceText(self, wordPosition, component, componentType):
        if(componentType == 'claim'):
            initialTag = '(I,claim)'
        else:
            initialTag = '(I,premise)'
        component.words[0].tag = initialTag
        for word in component.words:
            self.processedText[wordPosition] = word
            wordPosition += 1

    def processText(self):
        for claim in self.existingClaimsAndPremises.claims:
            wordPosition = 0

            for word in self.originalText:
                if (claim.words[0].content == word.content):
                    if (self.matchText(wordPosition, claim)):
                        self.replaceText(wordPosition, claim, 'claim')
                        claim.startPosition = wordPosition

                wordPosition += 1

        for premise in self.existingClaimsAndPremises.premises:
            wordPosition = 0
            for word in self.originalText:
                if (premise.words[0].content == word.content):
                    if (self.matchText(wordPosition, premise)):
                        self.replaceText(wordPosition, premise, 'premise')
                        premise.startPosition = wordPosition
                wordPosition += 1


class OutputWriter:
    def __init__(self, processedText, file):
        self.processedText = processedText
        self.textFile = open("OutputPunctuation/texts/" + file + '.txt', "w", encoding='utf-8')
        self.tagFile = open("OutputPunctuation/tags/" + file + '.txt', "w", encoding='utf-8')
        self.bothFile = open("OutputPunctuation/both/" + file + '.txt', "w", encoding='utf-8')

    def writeToTextFile(self):
        for word in self.processedText:
            content = word.content
            tag = word.tag
            self.textFile.write(u'' + content + '\n')
            self.tagFile.write(u'' + tag + '\n')
            self.bothFile.write(u'' + content + ' ' + tag + '\n')


class Pipeline:

    def translate(self):
        fileList = os.listdir("brat-project-final/ann")
        startTime = datetime.datetime.now().replace(microsecond=0)

        for file in fileList:
            fileName = re.sub('.ann', '', file)

            components = ReadComponents(fileName + '.ann')
            components.readComponents()

            text = ReadText(fileName + '.txt')
            text.readText()

            componentReplacer = ComponentsReplacer(text.words, components)
            componentReplacer.processText()

            output = OutputWriter(componentReplacer.processedText, fileName)
            output.writeToTextFile()

        endTime = datetime.datetime.now().replace(microsecond=0)
        timeTaken = endTime - startTime

        print("Isto demorou ")
        print(timeTaken)






pipeline = Pipeline()
pipeline.translate()


'''components = ReadComponents('essay021.ann')
components.readComponents()
#for component in components.text:
#    print(component.getTags())

text = ReadText('essay021.txt')
text.readText()

componentReplacer = ComponentsReplacer(text.words,components)
componentReplacer.processText()
print(componentReplacer.getTags())'''
