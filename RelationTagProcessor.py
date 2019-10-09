import os
class RelationTagProcessor:

    def __init__(self, baseDirectory):
        self.baseDirectory = baseDirectory
        self.component_tags = []
        self.distance_tags_list = []
        self.num_tags = 0
        self.nArgTag = None

        self.numNArg = 0
        self.numPremise = 0
        self.numClaim = 0

        self.tag_code = {}

    def processTags(self, fileName):
        file = open(fileName, "r", encoding='utf8')
        tag_contents = file.readlines()
        component = ''
        distance = []
        for tag in tag_contents:
            parts = tag.rsplit(',', 1)
            component += parts[0]  + ') '
            distance.append(parts[1][:-2])
        self.component_tags.append(component)
        self.distance_tags_list.append(distance)
        self.tag_count(tag_contents)

    def readTags(self):
        fileList = os.listdir(self.baseDirectory)
        for file in fileList:
            fileName = self.baseDirectory + '/' + file
            self.processTags(fileName)

    def tag_count(self, tags_list):
        for tag in tags_list:
            if tag == '':
                continue
            tag_elements = tag.split(',')
            if tag_elements[0][1] == 'O':
                self.numNArg += 1
            elif tag_elements[1] == 'premise':
                self.numPremise += 1
            elif tag_elements[1] == 'claim':
                self.numClaim += 1
