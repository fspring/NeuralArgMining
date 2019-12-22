import numpy as np
import copy

class Evaluator:
    def __init__(self, num_tags, num_measures, tags):
        self.num_tags = num_tags
        self.num_measures = num_measures
        self.tags = tags

    def empty_cvscores(self):
        #[acc, [precision[class], recall[class], f1[class]],[100_correct],[75_correct],[50_correct]]
        #percent_correct: [acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]
        empty_avg = list(map(float,np.zeros(self.num_tags)))
        # nr_measures = 1 + 3*(self.num_tags-1)
        empty_corr = list(map(float,np.zeros(self.num_measures)))
        cvscores = [0, [copy.deepcopy(empty_avg), copy.deepcopy(empty_avg), copy.deepcopy(empty_avg)],
                        copy.deepcopy(empty_corr), copy.deepcopy(empty_corr), copy.deepcopy(empty_corr)]

        return cvscores

    def spanCreator(self, unencodedY):
        spans = []
        illegal_i = 0
        for text in unencodedY:
            text = np.trim_zeros(text)
            textSpans = {}
            startPosition = 0
            currentPosition = 0
            lastTag = text[0]
            is_arg = False
            for tag in text:
                if tag == 3: # first premise token
                    is_arg = True
                    endPosition = currentPosition - 1
                    textSpans[startPosition] = endPosition
                    startPosition = currentPosition
                elif tag == 4: # first claim token
                    is_arg = True
                    endPosition = currentPosition - 1
                    textSpans[startPosition] = endPosition
                    startPosition = currentPosition
                elif tag == 2 and tag != lastTag: # first non-arg token
                    is_arg = False
                    endPosition = currentPosition - 1
                    textSpans[startPosition] = endPosition
                    startPosition = currentPosition
                elif tag == 1 and not is_arg: # invalid I tag
                    illegal_i += 1
                    if tag != lastTag:
                        endPosition = currentPosition - 1
                        textSpans[startPosition] = endPosition
                        startPosition = currentPosition
                lastTag = tag
                currentPosition += 1
            endPosition = currentPosition - 1
            textSpans[startPosition] = endPosition
            spans.append(textSpans)

        print('ILLEGAL I TAGS:', illegal_i) #debug
        return spans

    def spanEval(self, y_pred_class, unencodedY, threshold):
        goldSpans = self.spanCreator(unencodedY)
        empty = list(map(float,np.zeros(self.num_tags)))
        i = 0
        precision = copy.deepcopy(empty)
        recall = copy.deepcopy(empty)
        f1 = copy.deepcopy(empty)
        predictedSpanTypes = copy.deepcopy(empty) # predicted number of premises, claims and non-arg
        goldSpanTypes = copy.deepcopy(empty) # total number of premises, claims and non-arg
        precisionCorrectSpans = copy.deepcopy(empty)
        recallCorrectSpans = copy.deepcopy(empty)
        for result in y_pred_class: #for each text
            sequenceLength = len(np.trim_zeros(unencodedY[i]))
            result = np.resize(result, (sequenceLength, self.num_tags)) #(text size, n tags) -> [[0, 1, 0, 0], [0, 0, 1, 0], ...]
            classes = np.argmax(result, axis=1) #[1, 2, ...] -> size of text
            classes = np.add(classes, 1) #[2, 3, ...] -> to match unencodedY

            for spanStart, spanEnd in goldSpans[i].items():
                span_start_tag = unencodedY[i][spanStart]
                goldSpanTypes[span_start_tag - 1] += 1

            for spanStart, spanEnd in goldSpans[i].items():
                predicted = classes[spanStart:spanEnd + 1]
                possibleSpans = self.spanCreator([predicted])

                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    predicted_start_tag = classes[spanStart + possibleSpanStart]
                    predictedSpanTypes[predicted_start_tag - 1] += 1

                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    if (((possibleSpanEnd - possibleSpanStart + 1) >= ((spanEnd - spanStart + 1) * threshold))
                            and (classes[spanStart + possibleSpanStart] == unencodedY[i][
                                spanStart + possibleSpanStart])):
                        precisionCorrectSpans[classes[spanStart + possibleSpanStart] - 1] += 1
                        break

                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    if (((possibleSpanEnd - possibleSpanStart + 1) >= ((spanEnd - spanStart + 1) * threshold))
                            and (classes[spanStart + possibleSpanStart] == unencodedY[i][
                                spanStart + possibleSpanStart])):
                        recallCorrectSpans[classes[spanStart + possibleSpanStart] - 1] += 1
            i += 1

        precision_arg = 0
        goldSpanTypes_arg = 0
        for i in range(0, self.num_tags):
            precision_arg += precisionCorrectSpans[i]
            goldSpanTypes_arg += goldSpanTypes[i]
        accuracy = precision_arg / goldSpanTypes_arg

        for i in range(0, self.num_tags):
            if (predictedSpanTypes[i] != 0):
                precision[i] = (precisionCorrectSpans[i] / predictedSpanTypes[i])
            if (goldSpanTypes[i] != 0):
                recall[i] = (recallCorrectSpans[i] / goldSpanTypes[i])
            if ((precision[i] + recall[i]) != 0):
                f1[i] = 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i]))

        premise_index = self.tags.index('(P)')
        claim_index = self.tags.index('(C)')
        print('Accuracy at ' + str(threshold) + ' - ' + str(round(accuracy, 3)))
        print('Precision for premises at ' + str(threshold) + ' - ' + str(round(precision[premise_index], 3)))
        print('Precision for claims at ' + str(threshold) + ' - ' + str(round(precision[claim_index], 3)))
        print('Recall for premises at ' + str(threshold) + ' - ' + str(round(recall[premise_index], 3)))
        print('Recall for claims at ' + str(threshold) + ' - ' + str(round(recall[claim_index], 3)))
        print('F1 for premises at ' + str(threshold) + ' - ' + str(round(f1[premise_index], 3)))
        print('F1 for claims at ' + str(threshold) + ' - ' + str(round(f1[claim_index], 3)))

        ret = [round(accuracy, 4),round(precision[premise_index], 4),round(precision[claim_index], 4),round(recall[premise_index], 4),
            round(recall[claim_index], 4),round(f1[premise_index], 4),round(f1[claim_index], 4)]
        return ret

    def prettyPrintScore(self, score, scoreName):
        print(scoreName)
        numTexts = len(score)
        score_sum = np.zeros(self.num_tags)
        for scoreValue in score:
            score_sum += scoreValue[:self.num_tags]

        str_res = []
        float_res = []
        for i in range(0, self.num_tags):
            str_res.append(str(round(score_sum[i] / numTexts, 3)))
            float_res.append(round(score_sum[i] / numTexts, 4))

        print('\t'.join(self.tags))
        print('\t'.join(str_res))

        return float_res

    def tagEval(self, y_pred_class, unencodedY):
        i = 0
        precision = []
        recall = []
        f1 = []
        accuracy = []
        for result in y_pred_class: #for each text
            sequenceLength = len(np.trim_zeros(unencodedY[i])) #text size
            result = np.resize(result, (sequenceLength, self.num_tags)) #[[0, 1, 0, 0], [0, 0, 1, 0], ...]
            classes = np.argmax(result, axis=1) #[1, 2, ...]
            accuracy.append(accuracy_score(np.trim_zeros(unencodedY[i]), np.add(classes, 1)))
            scores = precision_recall_fscore_support(np.trim_zeros(unencodedY[i]), np.add(classes, 1))
            precision.append(np.pad(scores[0], (0,(self.num_tags - len(scores[0]))), 'constant'))
            recall.append(np.pad(scores[1], (0,(self.num_tags - len(scores[0]))), 'constant'))
            f1.append(np.pad(scores[2], (0,(self.num_tags - len(scores[0]))), 'constant'))
            i += 1
        print("Accuracy = %.3f%% (+/- %.3f%%)" % (np.mean(accuracy), np.std(accuracy)))
        precision = self.prettyPrintScore(precision, 'Precision')
        recall = self.prettyPrintScore(recall, 'Recall')
        f1 = self.prettyPrintScore(f1, 'F1')

        return [precision, recall, f1]

    def dist_eval(self, y_pred_dist, y_test_dist, y_test_class, unencodedY):
        nr_files = len(y_test_dist)
        tp = [0]*self.num_tags
        n = [0]*self.num_tags
        for i in range(0, nr_files):
            text_size = len(np.trim_zeros(unencodedY[i]))
            for j in range(0, text_size):
                pred = round(y_pred_dist[i][j][0])
                tag = np.argmax(y_test_class[i][j])
                n[tag] += 1
                if int(pred) == int(y_test_dist[i][j][0]):
                    tp[tag] += 1
        for i in range(0, self.num_tags):
            print('======', self.tags[i], '======')
            print('Correct distances:', tp[i], '------- Ratio', tp[i]/n[i], '\n')
        return (tp, n)

    def handleScores(self, oldScores, newScores, nFolds):
        newAccuracy = oldScores[0] + (newScores[0] / nFolds)
        empty = list(map(float,np.zeros(self.num_tags)))

        #[precision[class], recall[class], f1[class]]
        newTagScores = [copy.deepcopy(empty), copy.deepcopy(empty), copy.deepcopy(empty)]
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]
        # nr_measures = 1 + 3*(self.num_tags-1)

        newSpanAt1Scores = list(map(float,np.zeros(self.num_measures)))
        newSpanAt075Scores = list(map(float,np.zeros(self.num_measures)))
        newSpanAt050Scores = list(map(float,np.zeros(self.num_measures)))

        for i in range(0, 3):
            # for j in range(0, self.num_tags):
                # newTagScores[i][j] = oldScores[1][i][j] + (newScores[1][i][j] / nFolds)
            newTagScores[i] = list(map(float,np.array(oldScores[1][i]) + (np.array(newScores[1][i]) / nFolds)))
        for j in range(0, self.num_measures):
            newSpanAt1Scores[j] = oldScores[2][j] + (newScores[2][j] / nFolds)
        for j in range(0, self.num_measures):
            newSpanAt075Scores[j] = oldScores[3][j] + (newScores[3][j] / nFolds)
        for j in range(0, self.num_measures):
            newSpanAt050Scores[j] = oldScores[4][j] + (newScores[4][j] / nFolds)

            #[acc, [precision[class], recall[class], f1[class]], --> avgs
            #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 100
            #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 75
            #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]]  --> at 50
        return [newAccuracy, newTagScores, newSpanAt1Scores, newSpanAt075Scores, newSpanAt050Scores]

    def prettyPrintResults(self, scores):

        #[acc, [precision[class], recall[class], f1[class]], --> avgs
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 100
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2],  --> at 75
        #[acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]]  --> at 50

        premise_index = self.tags.index('(P)')
        claim_index = self.tags.index('(C)')

        print('Accuracy - ' + str(round(scores[0], 4)))

        print('Accuracy at ' + str(1) + ' - ' + str(round(scores[2][0], 3)))
        print('Accuracy at ' + str(0.75) + ' - ' + str(round(scores[3][0], 3)))
        print('Accuracy at ' + str(0.5) + ' - ' + str(round(scores[4][0], 3)))

        print('Precision for premises at ' + str(1) + ' - ' + str(round(scores[2][1], 3)))
        print('Precision for claims at ' + str(1) + ' - ' + str(round(scores[2][2], 3)))
        print('Precision for premises at ' + str(0.75) + ' - ' + str(round(scores[3][1], 3)))
        print('Precision for claims at ' + str(0.75) + ' - ' + str(round(scores[3][2], 3)))
        print('Precision for premises at ' + str(0.5) + ' - ' + str(round(scores[4][1], 3)))
        print('Precision for claims at ' + str(0.5) + ' - ' + str(round(scores[4][2], 3)))

        print('Recall for premises at ' + str(1) + ' - ' + str(round(scores[2][3], 3)))
        print('Recall for claims at ' + str(1) + ' - ' + str(round(scores[2][4], 3)))
        print('Recall for premises at ' + str(0.75) + ' - ' + str(round(scores[3][3], 3)))
        print('Recall for claims at ' + str(0.75) + ' - ' + str(round(scores[3][4], 3)))
        print('Recall for premises at ' + str(0.5) + ' - ' + str(round(scores[4][3], 3)))
        print('Recall for claims at ' + str(0.5) + ' - ' + str(round(scores[4][4], 3)))

        print('F1 for premises at ' + str(1) + ' - ' + str(round(scores[2][5], 3)))
        print('F1 for claims at ' + str(1) + ' - ' + str(round(scores[2][6], 3)))
        print('F1 for premises at ' + str(0.75) + ' - ' + str(round(scores[3][5], 3)))
        print('F1 for claims at ' + str(0.75) + ' - ' + str(round(scores[3][6], 3)))
        print('F1 for premises at ' + str(0.5) + ' - ' + str(round(scores[4][5], 3)))
        print('F1 for claims at ' + str(0.5) + ' - ' + str(round(scores[4][6], 3)))

        print('Precision')
        print('\t'.join(self.tags))
        print(str(round(scores[1][0][0], 3)) + '   ' + str(round(scores[1][0][1], 3)) + '     ' + str(
            round(scores[1][0][2], 3)) + '   ' + str(round(scores[1][0][3], 3)))
        print('Recall')
        print('\t'.join(self.tags))
        print(str(round(scores[1][1][0], 3)) + '   ' + str(round(scores[1][1][1], 3)) + '     ' + str(
            round(scores[1][1][2], 3)) + '   ' + str(round(scores[1][1][3], 3)))
        print('F1')
        print('\t'.join(self.tags))
        print(str(round(scores[1][2][0], 3)) + '   ' + str(round(scores[1][2][1], 3)) + '     ' + str(
            round(scores[1][2][2], 3)) + '   ' + str(round(scores[1][2][3], 3)))
