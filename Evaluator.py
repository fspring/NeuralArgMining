import numpy as np
import copy

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
class Evaluator:
    def __init__(self, num_tags, num_measures, tags):
        self.num_tags = num_tags
        self.num_measures = num_measures
        self.tags = tags

    def empty_cvscores(self):
        #[acc, [precision[class], recall[class], f1[class]],[100_correct],[75_correct],[50_correct]]
        #percent_correct: [acc, precision_c1, precision_c2, recall_c1, recall_c2, f1_c1, f1_c2]
        empty_avg = list(map(float,np.zeros(self.num_tags - 1)))
        # nr_measures = 1 + 3*(self.num_tags-1)
        empty_corr = list(map(float,np.zeros(self.num_measures)))
        cvscores = [0, [copy.deepcopy(empty_avg), copy.deepcopy(empty_avg), copy.deepcopy(empty_avg)],
                        copy.deepcopy(empty_corr), copy.deepcopy(empty_corr), copy.deepcopy(empty_corr)]

        return cvscores

    def spanCreator(self, tags):
        spans = []
        illegal_i = 0
        for text in tags:
            text_spans = {}

            start_pos = -1
            current_pos = -1
            prev_tag = -1

            text_size = len(text)
            
            for i in range(0, text_size):
                tag = text[i]
                if tag == 1: #should not exist
                    illegal_i += 1
                elif tag == 2:
                    if tag != prev_tag: #first non-arg token
                        if start_pos >= 0 and current_pos >= 0 and prev_tag >= 0:
                            text_spans[start_pos] = current_pos
                        start_pos = i
                        current_pos = i
                        prev_tag = tag
                    else:
                        current_pos = i
                        prev_tag = tag
                elif tag == 3:
                    if tag != prev_tag: #first premise token
                        if start_pos >= 0 and current_pos >= 0 and prev_tag >= 0:
                            text_spans[start_pos] = current_pos
                        start_pos = i
                        current_pos = i
                        prev_tag = tag
                    else:
                        current_pos = i
                        prev_tag = tag
                elif tag == 4:
                    if tag != prev_tag: #first claim token
                        if start_pos >= 0 and current_pos >= 0 and prev_tag >= 0:
                            text_spans[start_pos] = current_pos
                        start_pos = i
                        current_pos = i
                        prev_tag = tag
                    else:
                        current_pos = i
                        prev_tag = tag

            spans.append(text_spans)
        if illegal_i > 0:
            print('ILLEGAL I TAGS:', illegal_i) #debug
        return spans

    def spanEval(self, pred_spans, true_spans, threshold):
        true_spans_dict_list = self.spanCreator(true_spans)
        pred_spans_dict_list = self.spanCreator(pred_spans)

        empty = list(map(float,np.zeros(self.num_tags)))
        
        precision = copy.deepcopy(empty)
        recall = copy.deepcopy(empty)
        f1 = copy.deepcopy(empty)

        selected_spans = copy.deepcopy(empty) # predicted number of premises and claims
        relevant_spans = copy.deepcopy(empty) # total number of premises and claims

        true_positives = copy.deepcopy(empty) # number of correctly predicted premises and claims
        
        for i in range(0, len(true_spans_dict_list)): #for each text
            true_spans_dict = true_spans_dict_list[i]
            pred_spans_dict = pred_spans_dict_list[i]
            iteration = 0
            for true_start, true_end in true_spans_dict.items():
                true_tag = true_spans[i][true_start]

                relevant_spans[true_tag - 1] += 1
                
                for pred_start, pred_end in pred_spans_dict.items():
                    pred_tag = pred_spans[i][pred_start]

                    if iteration == 0:
                        selected_spans[pred_tag - 1] += 1

                    if true_tag == 2 and true_start == true_end: #single non-arg token
                        if pred_start == true_start and pred_end == true_end and pred_tag == 2:
                            true_positives[true_tag - 1] += 1

                    elif true_start <= pred_start and pred_start <= true_end: #exists intersection
                        
                        if pred_end <= true_end: #pred span contained in true span
                            if pred_tag == true_tag: #same tag
                                span_intersection = pred_end - pred_start
                                span_union = true_end - true_start
                                overlap = span_intersection / span_union
                                if overlap >= threshold: #is correct
                                    true_positives[true_tag - 1] += 1
                        
                        elif pred_end > true_end: #pred span end exceeds true span
                            if pred_tag == true_tag: #same tag
                                span_intersection = true_end - pred_start
                                span_union = pred_end - true_start
                                overlap = span_intersection / span_union
                                if overlap >= threshold: #is correct
                                    true_positives[true_tag - 1] += 1

                    elif true_start > pred_start and true_start <= pred_end:  #exists intersection

                        if pred_end <= true_end: #pred span start exceeds true span
                            if pred_tag == true_tag: #same tag
                                span_intersection = pred_end - true_start
                                span_union = true_end - pred_start
                                overlap = span_intersection / span_union
                                if overlap >= threshold: #is correct
                                    true_positives[true_tag - 1] += 1

                        elif pred_end > true_end: #true span contained in pred span
                            if pred_tag == true_tag: #same tag
                                span_intersection = true_end - true_start
                                span_union = pred_end - pred_start
                                overlap = span_intersection / span_union
                                if overlap >= threshold: #is correct
                                    true_positives[true_tag - 1] += 1

                iteration += 1




        true_all = 0
        all_N = 0
        for i in range(0, self.num_tags):
            true_all += true_positives[i]
            all_N += relevant_spans[i]
        accuracy = true_all / all_N

        for i in range(0, self.num_tags):
            if (selected_spans[i] != 0):
                precision[i] = (true_positives[i] / selected_spans[i])
            if (relevant_spans[i] != 0):
                recall[i] = (true_positives[i] / relevant_spans[i])
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
        score_sum = np.zeros(self.num_tags-1)
        for scoreValue in score:
            score_sum += scoreValue[:self.num_tags-1]

        str_res = []
        float_res = []
        for i in range(0, self.num_tags-1):
            str_res.append(str(round(score_sum[i] / numTexts, 3)))
            float_res.append(round(score_sum[i] / numTexts, 4))

        print('\t'.join(self.tags[1:]))
        print('\t'.join(str_res))

        return float_res

    def tagEval(self, pred_spans, true_spans):
        i = 0
        
        precision = []
        recall = []
        f1 = []
        accuracy = []

        nr_texts = len(true_spans)
        nr_classes = self.num_tags - 1 #I tag removed after training
        
        for i in range(0, nr_texts): #for each text

            accuracy.append(accuracy_score(true_spans[i], pred_spans[i]))
            scores = precision_recall_fscore_support(true_spans[i], pred_spans[i])
            
            precision.append(np.pad(scores[0], (0,(nr_classes - len(scores[0]))), 'constant'))
            recall.append(np.pad(scores[1], (0,(nr_classes - len(scores[0]))), 'constant'))
            f1.append(np.pad(scores[2], (0,(nr_classes - len(scores[0]))), 'constant'))
            
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
        empty = list(map(float,np.zeros(self.num_tags - 1)))

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
        print('\t'.join(self.tags[1:]))
        print(str(round(scores[1][0][0], 3)) + '   ' + str(round(scores[1][0][1], 3)) + '     ' + str(
            round(scores[1][0][2], 3)))
        print('Recall')
        print('\t'.join(self.tags[1:]))
        print(str(round(scores[1][1][0], 3)) + '   ' + str(round(scores[1][1][1], 3)) + '     ' + str(
            round(scores[1][1][2], 3)))
        print('F1')
        print('\t'.join(self.tags[1:]))
        print(str(round(scores[1][2][0], 3)) + '   ' + str(round(scores[1][2][1], 3)) + '     ' + str(
            round(scores[1][2][2], 3)))
