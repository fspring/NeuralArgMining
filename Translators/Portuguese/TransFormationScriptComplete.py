import json
import re
import os
from bs4 import BeautifulSoup
import io
import html2text
#import transformationScript
import datetime

#from pprint import pprint

class Word:
    content = ""
    tag = ""

    # The class "constructor" - It's actually an initializer
    def __init__(self, content, tag):
        self.content = content
        self.tag = tag

class Component:
    distance = 20
    startPosition = -1
    position = -1

    def setDistance(self, distance):
        self.distance = distance
        self.startPosition = -1
        self.position = -1

class Premise(Component):
    words = []

    # The class "constructor" - It's actually an initializer
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
    words = []

    # The class "constructor" - It's actually an initializer
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


class Translator:

    contents = {}

    def __init__(self):
        self.contents = {}

    def addPair(self, htmlFile, jsonfile):
        self.contents[jsonfile] = htmlFile

    def createAssociation(self, nodeSet):
        fileName = "corpusInput/json/" + nodeSet
        file = open(fileName, "r")
        contents = file.read()
        elements = json.loads(contents)

        for node in elements['nodes']:
            if 'http' in node['text']:
                link = re.search("(?P<url>https?://[^\s]+)", node['text']).group("url")
                link = re.sub('http://web.fe.up.pt/~ei11124/argmine_news/', '', link)
                link = link[:-1]
                self.addPair(link,nodeSet)
                break


    def createAssociations(self):
        fileList = os.listdir("corpusInput/json")
        for file in fileList:
            self.createAssociation(file)

class TextDumper:
    _nArgTag = "(O,|)"

    words = []
    file = ""

    def __init__(self, htmlFile):
        self.file = "corpusInput/html/" + htmlFile + '.html'
        self.words = []

    def getText(self):
        words = []
        for word in self.words:
            words.append(word.content)
        return ' '.join(words)

    def stripHtml(self):
        with io.open(self.file, 'r', encoding='utf8') as f:
            contents = f.read()
        plainText = html2text.html2text(contents)
        sentences = plainText.split('\n')

        maxSize = sentenceNumber = chosen = 0
        for sentence in sentences:
            size = len(sentence)
            if(size > maxSize):
                chosen = sentenceNumber
            sentenceNumber += 1

        sentences[chosen] = re.sub(r'[.]+(?![0-9])', r' .', sentences[chosen])
        sentences[chosen] = re.sub(r'[:]+(?![0-9])', r' :', sentences[chosen])
        sentences[chosen] = re.sub(r'[,]+(?![0-9])', r' ,', sentences[chosen])
        sentences[chosen] = re.sub(r'[;]+(?![0-9])', r' ;', sentences[chosen])
        sentences[chosen] = re.sub(r'[?]+(?![0-9])', r' ?', sentences[chosen])
        sentences[chosen] = re.sub(r'[!]+(?![0-9])', r' !', sentences[chosen])
        sentences[chosen] = re.sub(r'[…]+(?![0-9])', r' …', sentences[chosen])
        sentences[chosen] = re.sub(r'[“]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[”]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'["]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[‘]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[’]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[(]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[)]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[\']+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[`]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[`]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[[]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[]]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[«]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[»]+(?![0-9])', r'', sentences[chosen])
        sentences[chosen] = re.sub(r'[**]+(?![0-9])', r'', sentences[chosen])


        print(sentences[chosen])
        return sentences[chosen]

    def wordifyText(self):
        text = self.stripHtml()
        originalWords = text.split(' ')
        for word in originalWords:
            '''if(word == '.'):
                taggedWord = Word(word, '.')
                self.words.append(taggedWord)
            elif(word != ''):
                taggedWord = Word(word, self._nArgTag)
                self.words.append(taggedWord)'''
            if (word != ''):
                taggedWord = Word(word, self._nArgTag)
                self.words.append(taggedWord)


class claimsAndPremises:
    claims = []
    premises = []
    premisesToClaims = {}
    file = ""

    def __init__(self, jsonFile):
        self.file = "corpusInput/json/" + jsonFile + '.json'
        self.claims = []
        self.premises = []
        self.premisesToClaims = {}

    def removeHttp(self, elements):
        for node in elements['nodes']:
            if 'http' in node['text']:
                elements['nodes'].remove(node)

        return elements

    def removeInferences(self, elements):
        for node in elements['nodes']:
            if 'Default Inference' in node['text']:
                elements['nodes'].remove(node)

        return elements

    def collapseEdges(self, edges, nodes):
        collapsedEdges = []
        for originEdge in edges:
            for destinationEdge in edges:
                if ((destinationEdge['fromID'] == originEdge['toID']) & (self.getNodeText(nodes,originEdge['fromID']) != self.getNodeText(nodes,destinationEdge['toID']))):
                    edge = {originEdge['fromID']:destinationEdge['toID']}
                    collapsedEdges.append(edge)
                    #collapsedEdges[originEdge['fromID']] = destinationEdge['toID']

        #print(collapsedEdges)
        return collapsedEdges

    def getNodeText(self, nodes, nodeId):
        nodeText = ''
        for node in nodes:
            if (node['nodeID'] == nodeId):
                nodeText = node['text']
                nodeText = re.sub(r'[.]+(?![0-9])', r' .', nodeText)
                nodeText = re.sub(r'[:]+(?![0-9])', r' :', nodeText)
                nodeText = re.sub(r'[,]+(?![0-9])', r' ,', nodeText)
                nodeText = re.sub(r'[;]+(?![0-9])', r' ;', nodeText)
                nodeText = re.sub(r'[?]+(?![0-9])', r' ?', nodeText)
                nodeText = re.sub(r'[!]+(?![0-9])', r' !', nodeText)
                nodeText = re.sub(r'[…]+(?![0-9])', r' …', nodeText)
                nodeText = re.sub(r'[“]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[”]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'["]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[‘]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[’]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[(]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[)]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[\']+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[`]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[`]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[[]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[]]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[«]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[»]+(?![0-9])', r'', nodeText)
                nodeText = re.sub(r'[**]+(?![0-9])', r'', nodeText)

        return nodeText

    def tagClaimOrPremise(self, words, type):
        if (type == 'premise'):
            distance = '1'
        else:
            distance = '|'

        taggedSentence = []
        for wordIndex in range(0, len(words)):
            word = words[wordIndex]
            if (wordIndex == 0):
                #tag = '(B,' + type + ',' + distance + ')'
                #tag = '(B,' + type + ')'
                tag = '(I,' + type + ')'
            elif ((word == '.') or (word == ':') or (word == ';') or (word == '?') or (word == '!')):
                #tag = '(O,|,|)'
                tag = '(O,|)'
                #tag = '.'
            else:
                #tag = '(I,' + type + ',' + distance + ')'
                tag = '(I,' + type + ')'
            taggedWord = Word(word,tag)
            taggedSentence.append(taggedWord)
        return taggedSentence

    def isIntermediatePremise(self, claim, connections):
        isIntermediate = False
        for connection in connections:
            if next(iter(connection)) == claim:
                isIntermediate = True

        return isIntermediate

    def getPremisesAndClaims(self):
        file = open(self.file, "r")
        contents = file.read()
        elements = self.removeHttp(json.loads(contents))
        #elements = self.removeInferences(elements)

        connections = self.collapseEdges(elements['edges'], elements['nodes'])
        #print(self.file)
        #print(connections)
        nodes = elements['nodes']

        for connection in connections:
            claim = self.getNodeText(nodes, connection[next(iter(connection))])
            claimWords = claim.split()
            taggedClaim = Claim(self.tagClaimOrPremise(claimWords, 'claim'))
            self.claims.append(taggedClaim)

            premise = self.getNodeText(nodes, next(iter(connection)))
            premiseWords = premise.split()
            taggedPremise = Premise(self.tagClaimOrPremise(premiseWords, 'premise'))
            self.premises.append(taggedPremise)
            #print(taggedPremise.getText())

            self.premisesToClaims[premise] = claim



class claimsReplacer:
    processedText = []
    originalText = []
    existingClaimsAndPremises = []

    def __init__(self, originalText, existingClaimsAndPremises):
        self.originalText = originalText
        self.existingClaimsAndPremises = existingClaimsAndPremises
        self.processedText = originalText

    def getOriginalText(self):
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

    def matchText(self, wordPosition, component, textSize):
        #print(textSize)
        isMatch = True
        for word in component.words:
            #print(word.content)
            if ((wordPosition >= textSize) or
                    (word.content.lower() != self.originalText[wordPosition].content.lower())):
                isMatch = False
                break
            wordPosition += 1

        return isMatch

    def replaceText(self, wordPosition, component):
        for word in component.words:
            self.processedText[wordPosition] = word
            wordPosition += 1

    def processText(self):
        #print(self.getOriginalText())
        for claim in self.existingClaimsAndPremises.claims:
            wordPosition = 0
            for word in self.originalText:
                if (claim.words[0].content.lower() == word.content.lower()):
                    if(self.matchText(wordPosition, claim, len(self.originalText))):
                        self.replaceText(wordPosition, claim)
                        claim.startPosition = wordPosition
                wordPosition += 1

        for premise in self.existingClaimsAndPremises.premises:
            wordPosition = 0
            for word in self.originalText:
                if (premise.words[0].content.lower() == word.content.lower()):
                    if(self.matchText(wordPosition, premise, len(self.originalText))):
                        self.replaceText(wordPosition, premise)
                        premise.startPosition = wordPosition
                wordPosition += 1


class DistanceCalculator:
    processedText = []
    claimsAndPremises = []

    def __init__(self, processedText, existingClaimsAndPremises):
        self.processedText = processedText
        self.claimsAndPremises = existingClaimsAndPremises

    def getKey(self, component):
        return component.startPosition

    def returnUniqueComponents(self, components):
        index = 0
        uniqueComponents = []
        nonUniqueComponents = []
        unique = True
        for component in components:
            for secondComponent in uniqueComponents:
                if component.getText() == secondComponent.getText():
                    unique = False
                    nonUniqueComponents.append(component)
            if (unique):
                #print(component.getText())
                uniqueComponents.append(component)
            index += 1

        return uniqueComponents

    def arrangeComponents(self):
        #claims = self.returnUniqueComponents(self.claimsAndPremises.claims)
        #premises = self.returnUniqueComponents(self.claimsAndPremises.premises)

        '''for claim in claims:
            print("claim - " + claim.getText())
        for premise in premises:
            print("premise - " + premise.getText())'''
        components = self.claimsAndPremises.claims + self.claimsAndPremises.premises
        components = self.returnUniqueComponents(components)
        components = sorted(components, key=self.getKey)

        position = 1
        for component in components:
            #print("component - " + component.getText())
            #print (component.startPosition)
            for claim in self.claimsAndPremises.claims:
                if claim.startPosition == component.startPosition:
                    claim.position = position
                    #print("premise " + premise.getText())


            for premise in self.claimsAndPremises.premises:
                if premise.startPosition == component.startPosition:
                    premise.position = position
                    #print(premise.position)
                    #print("premise " + premise.getText())

            position += 1


    def calculateDistances(self):
        index = 0
        for premise in self.claimsAndPremises.premises:
            distance = self.claimsAndPremises.claims[index].position - self.claimsAndPremises.premises[index].position
            premise.distance = distance
            #print(distance)
            index += 1

    def updatePremises(self):
        for premise in self.claimsAndPremises.premises:
            for word in premise.words:
                tag = list(word.tag)
                if tag[1] != 'O':
                    tag[len(tag)-2] = str(premise.distance)
                    tag = "".join(tag)
                    word.tag = tag


class OutputWriter:
    processedText = []
    textFile = ""
    tagFile = ""
    file = ""

    def __init__(self, processedText, file):
        self.processedText = processedText
        self.file = open("corpusOutputPunctuation/txt/" + file + '.txt', "w", encoding='utf-8')
        #self.textFile = open("corpusOutput/txt/textsWithSentences/" + file + '.txt', "w", encoding='utf-8')
        self.textFile = open("corpusOutputPunctuation/txt/texts/" + file + '.txt', "w", encoding='utf-8')
        self.tagFile = open("corpusOutputPunctuation/txt/tags/" + file + '.txt', "w", encoding='utf-8')

    def writeToTextFile(self):
        for word in self.processedText:
            content = word.content
            tag = word.tag
            self.textFile.write(u'' + content + '\n')
            self.tagFile.write(u'' + tag + '\n')
            self.file.write(u'' + content + '' + tag + '\n')


class Pipeline:

    def translate(self):
        translator = Translator()
        translator.createAssociations()
        files = translator.contents

        startTime = datetime.datetime.now().replace(microsecond=0)
        for (jsonFile, htmlFile) in files.items():
            htmlFile = re.sub('.html', '', htmlFile)
            jsonFile = re.sub('.json', '', jsonFile)


            dumper = TextDumper(htmlFile)
            dumper.wordifyText()

            claims = claimsAndPremises(jsonFile)
            claims.getPremisesAndClaims()

            replacer = claimsReplacer(dumper.words, claims)
            replacer.processText()

            #distanceCalculator = DistanceCalculator(replacer.processedText, replacer.existingClaimsAndPremises)
            #distanceCalculator.arrangeComponents()
            #distanceCalculator.calculateDistances()
            #distanceCalculator.updatePremises()

            #replacer = claimsReplacer(dumper.words, distanceCalculator.claimsAndPremises)
            #replacer.processText()

            output = OutputWriter(replacer.processedText, jsonFile)
            output.writeToTextFile()

        endTime = datetime.datetime.now().replace(microsecond=0)
        timeTaken = endTime - startTime

        print("Isto demorou ")
        print(timeTaken)






pipeline = Pipeline()
pipeline.translate()


