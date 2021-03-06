import json
import re
import os
from bs4 import BeautifulSoup
import io
import html2text
import datetime
import copy


class Word:
    id = ''
    content = ''
    tag = ''

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
    id = ''
    # The class "constructor" - It's actually an initializer
    def __init__(self, words, id):
        self.words = words
        self.id = id

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

    def getIdInt(self):
        return int(self.id)

    def getIdStr(self):
        return self.id


class Claim(Component):
    words = []
    id = ''

    # The class "constructor" - It's actually an initializer
    def __init__(self, words, id):
        self.words = words
        self.id = id

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

    def getIdInt(self):
        return int(self.id)

    def getIdStr(self):
        return self.id


class Translator:

    contents = {}

    def __init__(self):
        self.contents = {}

    def addPair(self, htmlFile, jsonfile):
        self.contents[jsonfile] = htmlFile

    def createAssociation(self, nodeSet):
        fileName = 'corpusInput/json/' + nodeSet
        file = open(fileName, 'r')
        contents = file.read()
        elements = json.loads(contents)

        for node in elements['nodes']:
            if 'http' in node['text']:
                link = re.search('(?P<url>https?://[^\s]+)', node['text']).group('url')
                link = re.sub('http://web.fe.up.pt/~ei11124/argmine_news/', '', link)
                link = link[:-1]
                self.addPair(link,nodeSet)
                break


    def createAssociations(self):
        fileList = os.listdir('corpusInput/json')
        for file in fileList:
            self.createAssociation(file)

class TextDumper:
    _nArgTag = '(O,|)'

    words = [] #list of Word objects
    file = ''

    def __init__(self, htmlFile):
        self.file = 'corpusInput/html/' + htmlFile + '.html'
        self.words = []

    def getText(self):
        words = []
        for word in self.words:
            words.append(word.content)
        return ' '.join(words)

    def stripHtml(self):
        with io.open(self.file, 'r', encoding='utf8') as f:
            contents = f.read()

        # plainText = html2text.html2text(contents)
        if 'h2' in contents:
            print('h2')
        if 'h3' in contents:
            print('h3')
        soup = BeautifulSoup(contents,features="html.parser")
        soup.h1.decompose()
        plainText = soup.getText()
        # print(plainText)
        # sentences = plainText.split('\n')
        sentences = re.sub(r'\n', r' ', plainText)
        sentences = sentences.split('    ')

        whitespace = re.compile('\s')
        i = 0
        for sentence in sentences[:]:
            size = len(sentence)
            # print(i, sentence)
            if size == 0:
                # print('removed empty')
                sentences.remove(sentence)
            elif '#' in sentence:
                # print('removed header')
                sentences.remove(sentence)
            else:
                # print('cleaned and next')
                sentences[i] = re.sub(r'[.]+(?![0-9])', r' .', sentences[i])
                sentences[i] = re.sub(r'[:]+(?![0-9])', r' :', sentences[i])
                sentences[i] = re.sub(r'[,]+(?![0-9])', r' ,', sentences[i])
                sentences[i] = re.sub(r'[;]+(?![0-9])', r' ;', sentences[i])
                sentences[i] = re.sub(r'[?]+(?![0-9])', r' ?', sentences[i])
                sentences[i] = re.sub(r'[!]+(?![0-9])', r' !', sentences[i])
                sentences[i] = re.sub(r'[…]+(?![0-9])', r' … ', sentences[i])
                sentences[i] = re.sub(r'[\s]+(?![0-9])', r' ', sentences[i])
                sentences[i] = re.sub(r'[“]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[”]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'["]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[‘]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[’]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[(]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[)]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[\']+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[`]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[`]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[[]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[]]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[«]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[»]+(?![0-9])', r'', sentences[i])
                # sentences[i] = re.sub(r'[#]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'[**]+(?![0-9])', r'', sentences[i])
                sentences[i] = re.sub(r'\s', r' ', sentences[i])
                sentences[i] = re.sub(r'  ', r' ', sentences[i])
                i += 1

        separator = ' '
        full_sentence = separator.join(sentences)
        full_sentence = re.sub(r'([a-z])- ', r'\1-', full_sentence)
        return full_sentence

    #creates list of words with default tag (O,|)
    def wordifyText(self):
        text = self.stripHtml()
        originalWords = text.split(' ')
        for word in originalWords:
            if (word != ''):
                taggedWord = Word(word, self._nArgTag)
                self.words.append(taggedWord)


class claimsAndPremises:
    claims = []
    premises = []
    premisesToClaims = {}
    ordered_args = []
    premises_id = []
    claims_id = []
    close_connections = {}
    file = ''

    def __init__(self, jsonFile):
        self.file = 'corpusInput/json/' + jsonFile + '.json'
        self.claims = []
        self.premises = []
        self.premisesToClaims = {}
        self.ordered_args = []
        self.premises_id = []
        self.claims_id = []

    def removeHttp(self, elements):
        for node in elements['nodes']:
            if 'http' in node['text']:
                elements['nodes'].remove(node)

        return elements

    #removes links to Default Inference (RA) and Default Conflict (CA) nodes
    def remove_default_edges(self, edges, nodes):
        collapsedEdges = {}
        # print(edges)
        for i in range(0, len(edges)):
            for j in range(0, len(edges)):
                # if edges[i]['fromID'] == '263910':
                #     print('e1 ', edges[i])
                #     print('e2 ', edges[j], '\n')
                if edges[i]['toID'] == edges[j]['fromID'] and self.isDefaultNode(nodes, edges[i]['toID']):
                    if edges[i]['fromID'] in collapsedEdges.keys():
                        collapsedEdges[edges[i]['fromID']].append(edges[j]['toID'])
                    else:
                        collapsedEdges[edges[i]['fromID']] = [edges[j]['toID']]


        # print(collapsedEdges)
        return collapsedEdges

    def isDefaultNode(self,nodes,nodeId):
        nodeType = self.getNodeType(nodes,nodeId)
        if nodeType == 'CA' or nodeType == 'RA':
            return True
        else:
            return False

    def getNodeType(self, nodes, nodeId):
        for node in nodes:
            if(node['nodeID'] == nodeId):
                return node['type']

    #not currently in use to compare two nodes, and idk why it ever was
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
                nodeText = re.sub(r'[…]+(?![0-9])', r' … ', nodeText)
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
                nodeText = re.sub(r'  ', r' ', nodeText)

        return nodeText

    def tagClaimOrPremise(self, words, type, distance):
        taggedSentence = []
        for wordIndex in range(0, len(words)):
            word = words[wordIndex]
            if (wordIndex == 0):
                tag = '(' + type + ',' + distance + ')'
            elif wordIndex == len(words) - 1 and ((word == '.') or (word == ':') or (word == ';') or (word == '?') or (word == '!')):
                tag = '(O,|)'
            else:
                tag = '(I,' + distance + ')'
            taggedWord = Word(word,tag)
            taggedSentence.append(taggedWord)
        return taggedSentence

    def getPremisesAndClaims(self):
        file = open(self.file, 'r')
        contents = file.read()
        elements = self.removeHttp(json.loads(contents))
        connections = self.remove_default_edges(elements['edges'], elements['nodes'])
        # args = self.split_claims_from_premises(connections)
        # print(connections)
        self.orderArgComponents(connections)
        self.closeConnections(connections)
        nodes = elements['nodes']
        claim = premise = None
        # print(self.ordered_args)
        for id in self.ordered_args:
            if id in self.claims_id:
                # print('claim', id)
                claim = self.getNodeText(nodes, id)
                claimWords = claim.split()
                taggedClaim = Claim(self.tagClaimOrPremise(claimWords, 'C','0'), id)
                self.claims.append(taggedClaim)
            elif id in self.premises_id:
                # print('premise', id)
                premise = self.getNodeText(nodes, id)
                premiseWords = premise.split()
                taggedPremise = Premise(self.tagClaimOrPremise(premiseWords, 'P', '0'), id)
                self.premises.append(taggedPremise)

            self.premisesToClaims[premise] = claim

    def orderArgComponents(self,connections):
        values = connections.values()
        keys = connections.keys()
        for value in values:
            for id in value:
                if (id not in keys) and (id not in self.claims_id):
                    self.claims_id.append(id)
        for key in keys:
            if (key not in self.premises_id):
                self.premises_id.append(key)
        self.premises_id.sort()
        self.claims_id.sort()

        i = j = 0
        while i < len(self.claims_id) or j < len(self.premises_id):
            # print(i,',', j)
            if i == len(self.claims_id):
                self.ordered_args.append(self.premises_id[j])
                j += 1
            elif j == len(self.premises_id):
                self.ordered_args.append(self.claims_id[i])
                i += 1
            elif int(self.claims_id[i]) < int(self.premises_id[j]):
                self.ordered_args.append(self.claims_id[i])
                i += 1
            elif i == len(self.premises_id) or int(self.claims_id[i]) > int(self.premises_id[j]):
                self.ordered_args.append(self.premises_id[j])
                j += 1

    #adds the closure of each node to the existing connections
    def closeConnections(self, connections):
        close = {}
        values = connections.values()
        keys = connections.keys()
        for node in keys:
            queue = copy.deepcopy(connections[node])
            close[node] = set()
            while len(queue) > 0:
                link = queue[-1]
                close[node].add(link)
                if link not in close.keys():
                    close[link] = set([node])
                elif node not in close[link]:
                    close[link].add(node)
                queue.pop()
                if link in keys:
                    for el in connections[link]:
                        queue.append(el)
        self.close_connections = close

class Tag_Replacer:
    processedText = [] #list of Word objects
    originalText = [] #list of Word objects
    existingClaimsAndPremises = None #claimsAndPremises object

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

    def isMatch(self, wordPosition, component, textSize):
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

    #Replaces original Word object with new Word comtaining the processed tags and id
    def replaceText(self, wordPosition, component):
        for word in component.words:
            word.id = component.id
            self.processedText[wordPosition] = word
            wordPosition += 1

    #For each Word in Arg Component, finds match in original text with empty tags and replaces
    def processText(self):
        for claim in self.existingClaimsAndPremises.claims:
            wordPosition = 0
            for word in self.originalText:
                if (claim.words[0].content.lower() == word.content.lower()):
                    if(self.isMatch(wordPosition, claim, len(self.originalText))):
                        claim_copy = copy.deepcopy(claim) #because is compound object
                        self.replaceText(wordPosition, claim_copy)
                wordPosition += 1

        for premise in self.existingClaimsAndPremises.premises:
            wordPosition = 0
            for word in self.originalText:
                if (premise.words[0].content.lower() == word.content.lower()):
                    if(self.isMatch(wordPosition, premise, len(self.originalText))):
                        prem_copy = copy.deepcopy(premise) #because is compound object
                        self.replaceText(wordPosition, prem_copy)
                wordPosition += 1


class DistanceCalculator:
    close_connections = {} #dictionary {id: [id1, id2, ...]}, where the ids are all strings
    components_words = [] #list of Word objects

    def __init__(self, components_words, close_connections):
        self.components_words = components_words
        self.close_connections = close_connections

    #for each premise, finds the next arg component that is linked, and replaces distance in tag
    #only looks ahead --> distance always greater than or equal to 0
    def all_positive_distance_simple(self):
        text_size = len(self.components_words)
        for i in range(0, text_size):
            if self.components_words[i].id == '':
                continue
            id = self.components_words[i].id
            tag = self.components_words[i].tag
            tag_parts = tag.split(',')
            if tag_parts[1] != 'premise':
                continue
            if id not in self.close_connections.keys():
                continue
            linked_ids = self.close_connections[id]
            dist = 0
            for j in range(i+1, len(self.components_words)):
                dist += 1
                if self.components_words[j].id in linked_ids:
                    break
            if dist >= text_size: #this has never happened yet
                print('overflow')
                dist = 0
            tag = tag_parts[0] + ',' + tag_parts[1] + ',' + str(dist) + ')'
            self.components_words[i].tag = tag

    #for each premise and claim, finds the next arg component that is linked, and replaces distance in tag
    #only looks ahead --> distance always greater than or equal to 0
    def all_positive_distance_claims(self):
        text_size = len(self.components_words)
        # print(text_size)
        for i in range(0, text_size):
            # print('id', self.components_words[i].id)
            if self.components_words[i].id == '':
                continue
            id = self.components_words[i].id
            src_tag = self.components_words[i].tag
            # print('src_tag', src_tag)
            src_tag_parts = src_tag.split(',')
            # print(src_tag_parts)
            if src_tag_parts[0] == '(O':
                dist = 0
                continue
            if id not in self.close_connections.keys():
                dist = 0
                continue
            linked_ids = self.close_connections[id]
            if src_tag_parts[0] == '(I':
                if dist != 0:
                    dist -= 1
                # print(src_tag, dist)
                tag = src_tag_parts[0] + ',' + str(dist) + ')'
                # print('new',tag)
                self.components_words[i].tag = tag
                continue
            dist = 0
            is_linked = False
            for j in range(i+1, len(self.components_words)):
                dist += 1
                if self.components_words[j].id in linked_ids:
                    tgt_tag = self.components_words[j].tag
                    tgt_arg = tgt_tag.split(',')[0]
                    # print('tgt_tag', tgt_tag)
                    if src_tag_parts[0] == '(P' and tgt_arg == '(C':
                        is_linked = True
                        # print('yes premise-claim')
                        break
                    elif src_tag_parts[0] == '(P' and tgt_arg == '(P':
                        is_linked = True
                        # print('yes premise-premise')
                        break
                    elif src_tag_parts[0] == '(C' and tgt_arg == '(P':
                        is_linked = True
                        # print('yes claim-premise')
                        break
            if not is_linked:
                dist = 0
                continue
            elif dist >= text_size: #this has never happened yet
                print('overflow')
                dist = 0
            # print(src_tag, dist)
            tag = src_tag_parts[0] + ',' + str(dist) + ')'
            # print('new',tag)
            self.components_words[i].tag = tag

class OutputWriter:
    arg_components = [] #list of Word objects in order of original text
    textFile = ''
    tagFile = ''
    file = ''
    distance_calculator = None

    def __init__(self, arg_components, file):
        self.arg_components = arg_components
        if not os.path.exists('rel/texts/'):
            os.makedirs('rel/texts/')
        if not os.path.exists('rel/tags/'):
            os.makedirs('rel/tags/')
        self.file = open('rel/' + file + '.txt', 'w', encoding='utf-8')
        self.textFile = open('rel/texts/' + file + '.txt', 'w', encoding='utf-8')
        self.tagFile = open('rel/tags/' + file + '.txt', 'w', encoding='utf-8')

    def writeToTextFile(self):
        for word in self.arg_components:
            id = word.id
            content = word.content
            tag = word.tag
            self.textFile.write(u'' + content + '\n')
            self.tagFile.write(u'' + tag + '\n')
            self.file.write(u'' + content + '\t' + tag + '\n')

        self.textFile.close()
        self.tagFile.close()
        self.file.close()


class Pipeline:

    def translate(self):
        translator = Translator()
        translator.createAssociations()
        files = translator.contents

        startTime = datetime.datetime.now().replace(microsecond=0)
        for (jsonFile, htmlFile) in files.items():
            htmlFile = re.sub('.html', '', htmlFile)
            jsonFile = re.sub('.json', '', jsonFile)

            # if(htmlFile != '10'):
            #     continue
            # if(jsonFile != 'nodeset4699'):
            #     continue

            print(htmlFile)
            print(jsonFile)

            #HTML to plain text
            dumper = TextDumper(htmlFile)
            dumper.wordifyText()
            # print(dumper.getText())

            #identify claims and premises and tag them
            arg_components = claimsAndPremises(jsonFile)
            arg_components.getPremisesAndClaims()
            # for claim in arg_components.claims:
            #     assert claim.getText().lower() in dumper.getText().lower()
            #     print(claim.getText())
            # for prem in arg_components.premises:
            #     assert prem.getText().lower() in dumper.getText().lower()
            #     print(prem.getText())
            # print(len(arg_components.premises))

            #replace original text with tagged text
            replacer = Tag_Replacer(dumper.words, arg_components)
            replacer.processText()

            #calculate distance between related components and update tags
            distance_calculator = DistanceCalculator(replacer.processedText, arg_components.close_connections)
            distance_calculator.all_positive_distance_claims()

            #write to 3 kinds of output files: id + text + tags, id + text, id + tags
            output = OutputWriter(distance_calculator.components_words, jsonFile)
            output.writeToTextFile()
            # break
        endTime = datetime.datetime.now().replace(microsecond=0)
        timeTaken = endTime - startTime

        print('Isto demorou ')
        print(timeTaken)


pipeline = Pipeline()
pipeline.translate()
