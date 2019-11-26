import re
import os
import datetime
import copy

class Word:
    def __init__(self, content, tag, id):
        self.content = content
        self.tag = tag
        self.component_id = id

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
    def __init__(self, filename):
        self.claims = []
        self.premises = []
        self.relations = {}
        self.majorclaims_ids = []
        self.closure_relations = {}
        self.filename = "brat-project-final/" + filename

    def read_components(self):
        with open(self.filename, "r", encoding='utf8') as file:
            for line in file:
                line = re.sub(r'[\n]+(?![0-9])', r'', line)
                line = re.sub(r'[.]+(?![0-9])', r' .', line)
                line = re.sub(r'[:]+(?![0-9])', r' :', line)
                line = re.sub(r'[,]+(?![0-9])', r' ,', line)
                line = re.sub(r'[;]+(?![0-9])', r' ;', line)
                line = re.sub(r'[?]+(?![0-9])', r' ?', line)
                line = re.sub(r'[!]+(?![0-9])', r' !', line)

                if self.is_argcomponent(line):
                    self.normalize_structures(line)
                elif self.is_attribute(line):
                    continue #ignore
                else:
                    self.add_relation(line)
            self.relations_closure()
            self.add_majorclaim_relations()

    def is_argcomponent(self, line):
        comp_tag = line.split('\t')[1]
        component = comp_tag.split(" ", 1)[0]
        if component == 'MajorClaim':
            return True
        elif component == 'Premise':
            return True
        elif component == 'Claim':
            return True
        else:
            return False

    def is_attribute(self, line):
        id_tag = line.split('\t')[0]
        if id_tag[0] == 'A':
            return True
        else:
            return False

    def normalize_structures(self, line):
        tag_parts = line.split('\t')
        id = tag_parts[0]
        componentType = tag_parts[1].split(" ", 1)[0]
        sentence = tag_parts[2].split(" ")

        if ((componentType == 'MajorClaim') or (componentType == 'Claim')):
            tag = '(C,0)'
        else:
            tag = '(P,0)'

        taggedWords = []
        for i in range(0, len(sentence)):
            if i != 0:
                tag = '(I,0)'
            taggedWords.append(Word(sentence[i], tag, id))
        if componentType == 'MajorClaim':
            self.majorclaims_ids.append(id)
            self.claims.append(Claim(taggedWords))
        elif componentType == 'Claim':
            self.claims.append(Claim(taggedWords))
        else:
            self.premises.append(Premise(taggedWords))

    def add_relation(self, line):
        relation_tag = line.split('\t')[1]
        rel_parts = relation_tag.split(' ')
        rel_type = rel_parts[0]
        if(rel_type == 'Stance'):
            return
        arg1_id = rel_parts[2][1:]
        arg2_id = rel_parts[4][1:]
        self.relations[arg1_id] = arg2_id

    def add_majorclaim_relations(self):
        premises = self.relations.keys()
        for premise in premises:
            self.closure_relations[premise] += self.majorclaims_ids
        for mc in self.majorclaims_ids:
            self.closure_relations[mc] = premises


    def relations_closure(self):
        values = self.relations.values()
        keys = self.relations.keys()
        for node in keys:
            linked = self.relations[node]
            self.closure_relations[node] = [linked]
            if linked not in self.closure_relations.keys():
                self.closure_relations[linked] = [node]
            elif node not in self.closure_relations[linked]:
                self.closure_relations[linked].append(node)
            while linked in keys:
                linked = self.relations[linked]
                if linked not in self.closure_relations[node]:
                    self.closure_relations[node].append(linked)


class ReadText():
    def __init__(self, filename):
        self.words =  []
        self.filename = "brat-project-final/" + filename

    def read_text(self):
        line_nr = 1
        with open(self.filename, "r", encoding='utf8') as file:
            for line in file:
                line = re.sub(r'[\n]+(?![0-9])', r'', line)
                line = re.sub(r'[.]+(?![0-9])', r' .', line)
                line = re.sub(r'[:]+(?![0-9])', r' :', line)
                line = re.sub(r'[,]+(?![0-9])', r' ,', line)
                line = re.sub(r'[;]+(?![0-9])', r' ;', line)
                line = re.sub(r'[?]+(?![0-9])', r' ?', line)
                line = re.sub(r'[!]+(?![0-9])', r' !', line)
                if ((line_nr == 1) or (line_nr == 2)):
                    line_nr = line_nr + 1
                    continue
                else:
                    for word in line.split(" "):
                        if(len(word) > 0):
                            self.words.append(Word(word,'(O,|)', ''))


class ComponentsReplacer():
    def __init__(self, original_text, argument_components):
        self.original_text = original_text
        self.argument_components = argument_components
        self.processed_text = original_text

    def getText(self):
        words = []
        for word in self.original_text:
            words.append(word.content)
        return ' '.join(words)

    def getProcessedText(self):
        words = []
        for word in self.processed_text:
            words.append(word.content)
        return ' '.join(words)

    def getTags(self):
        tags = []
        for word in self.processed_text:
            tags.append(word.tag)
        return ' '.join(tags)


    def is_match(self, word_pos, component):
        for word in component.words:
            if (word_pos+1 > len(self.original_text)):
                return False
            elif (word.content != self.original_text[word_pos].content):
                return False
            word_pos += 1

        return True

    def replace_text(self, word_pos, component, componentType):
        for word in component.words:
            self.processed_text[word_pos] = word
            word_pos += 1

    def process_text(self):
        for claim in self.argument_components.claims:
            word_pos = 0

            for word in self.original_text:
                if (claim.words[0].content == word.content):
                    if (self.is_match(word_pos, claim)):
                        claim_copy = copy.deepcopy(claim) #because is compound object
                        self.replace_text(word_pos, claim_copy, 'claim')
                        claim.startPosition = word_pos

                word_pos += 1

        for premise in self.argument_components.premises:
            word_pos = 0
            for word in self.original_text:
                if (premise.words[0].content == word.content):
                    if (self.is_match(word_pos, premise)):
                        prem_copy = copy.deepcopy(premise) #because is compound object
                        self.replace_text(word_pos, prem_copy, 'premise')
                        prem_copy.startPosition = word_pos
                word_pos += 1

class DistanceCalculator:
    closure_relations = {} #dictionary {id: [id1, id2]}, where the ids are all strings
    words = [] #list of Word objects

    def __init__(self, words, closure_relations):
        self.words = words
        self.closure_relations = closure_relations

    #for each premise, finds the next arg component that is linked, and replaces distance in tag
    #only looks ahead --> distance always greater than or equal to 0
    def all_positive_distance_simple(self):
        text_size = len(self.words)
        for i in range(0, text_size):
            if self.words[i].component_id == '':
                continue
            id = self.words[i].component_id
            tag = self.words[i].tag
            tag_parts = tag.split(',')
            if tag_parts[1] != 'premise':
                continue
            if id not in self.closure_relations.keys():
                continue
            linked_ids = self.closure_relations[id]
            dist = 0
            for j in range(i+1, len(self.words)):
                dist += 1
                if self.words[j].component_id in linked_ids:
                    break
            if dist >= text_size: #this has never happened yet
                print('overflow')
                dist = 0
            tag = tag_parts[0] + ',' + tag_parts[1] + ',' + str(dist) + ')'
            self.words[i].tag = tag

    #for each premise and claim, finds the next arg component that is linked, and replaces distance in tag
    #only looks ahead --> distance always greater than or equal to 0
    def all_positive_distance_claims(self):
        text_size = len(self.words)
        for i in range(0, text_size):
            if self.words[i].component_id == '':
                continue
            id = self.words[i].component_id
            src_tag = self.words[i].tag
            # print('src_tag', src_tag)
            src_tag_parts = src_tag.split(',')
            # print(src_tag_parts)
            if src_tag_parts[0] == '(O':
                dist = 0
                continue
            if id not in self.closure_relations.keys():
                dist = 0
                continue
            linked_ids = self.closure_relations[id]
            if src_tag_parts[0] == '(I':
                if dist != 0:
                    dist -= 1
                # print(src_tag, dist)
                tag = src_tag_parts[0] + ',' + str(dist) + ')'
                # print('new',tag)
                self.words[i].tag = tag
                continue
            dist = 0
            is_linked = False
            for j in range(i+1, len(self.words)):
                dist += 1
                if self.words[j].component_id in linked_ids:
                    tgt_tag = self.words[j].tag
                    tgt_arg = tgt_tag.split(',')[0]
                    # print('tgt_tag', tgt_tag)
                    if src_tag_parts[0] == '(P' and tgt_arg == '(C':
                        is_linked = True
                        # print('yes premise-claim')
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
            # print(tag, dist)
            tag = src_tag_parts[0] + ',' + str(dist) + ')'
            # print('new',tag)
            self.words[i].tag = tag

class OutputWriter:
    def __init__(self, processed_text, file):
        self.processed_text = processed_text
        if not os.path.exists('OutputPunctuation/texts/'):
            os.makedirs('OutputPunctuation/texts/')
        if not os.path.exists('OutputPunctuation/tags/'):
            os.makedirs('OutputPunctuation/tags/')
        if not os.path.exists('OutputPunctuation/both/'):
            os.makedirs('OutputPunctuation/both/')
        self.textFile = open("OutputPunctuation/texts/" + file + '.txt', "w", encoding='utf-8')
        self.tagFile = open("OutputPunctuation/tags/" + file + '.txt', "w", encoding='utf-8')
        self.bothFile = open("OutputPunctuation/both/" + file + '.txt', "w", encoding='utf-8')

    def write_to_text_file(self):
        for word in self.processed_text:
            content = word.content
            tag = word.tag
            id = word.component_id
            self.textFile.write(u'' + content + '\n')
            self.tagFile.write(u'' + tag + '\n')
            self.bothFile.write(u'' + content + '\t' + tag + '\n')


class Pipeline:

    def translate(self):
        fileList = os.listdir("brat-project-final/ann")
        startTime = datetime.datetime.now().replace(microsecond=0)

        for file in fileList:
            filename = re.sub('.ann', '', file)

            # print(filename)

            components = ReadComponents(filename + '.ann')
            components.read_components()

            text = ReadText(filename + '.txt')
            text.read_text()

            component_replacer = ComponentsReplacer(text.words, components)
            component_replacer.process_text()

            #calculate distance between related components and update tags
            distance_calculator = DistanceCalculator(component_replacer.processed_text, components.closure_relations)
            distance_calculator.all_positive_distance_claims()

            output = OutputWriter(component_replacer.processed_text, filename)
            output.write_to_text_file()

            # break
        endTime = datetime.datetime.now().replace(microsecond=0)
        timeTaken = endTime - startTime

        print("Isto demorou ")
        print(timeTaken)


pipeline = Pipeline()
pipeline.translate()
