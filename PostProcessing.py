import numpy as np

class PostProcessing:
    def __init__(self, num_tags, tags):
        self.num_tags = num_tags
        self.tags = tags

    def correct_dist_prediction(self, arg_pred, dist_pred, unencodedY):
        print('=========== CORRECTING ===========') #debug
        f = open('correction_debug.txt', 'w')
        for i in range(0, len(dist_pred)):
            f.write(u'i: ' + str(i) + '\n')
            is_premise = False
            is_claim = False
            is_beginning = False
            text_size = len(np.trim_zeros(unencodedY[i]))
            for j in range(0, text_size): #ensure dist points to first token in arg comp or zero
                f.write(u'orig: ' + str(arg_pred[i][j]) + '\t' + str(dist_pred[i][j]) + '\n')
                src_arg = np.argmax(arg_pred[i][j])
                pred_dist = int(round(dist_pred[i][j][0]))
                if src_arg == 1: #non-arg
                    if is_beginning:
                        arg_pred[i][j][1] = 0 #erase O tag
                        arg_pred[i][j][0] = 1 #add I tag
                        dist_pred[i][j][0] = 0
                    is_premise = False
                    is_claim = False
                    is_beginning = False
                    dist_pred[i][j][0] = 0
                elif src_arg == 0 and not is_claim and not is_premise: #impossible inside tag
                    arg_pred[i][j][0] = 0 #erase I tag
                    arg_pred[i][j][1] = 1 #add O tag
                    dist_pred[i][j][0] = 0
                elif src_arg == 2 or (src_arg == 0 and is_premise): #premise
                    if src_arg == 2:
                        is_beginning = True
                    elif src_arg == 0:
                        is_beginning = False

                    is_premise = True
                    is_claim = False
                    if pred_dist == 0:
                        dist_pred[i][j][0] = 0
                        continue
                    tgt_index = j + pred_dist
                    if tgt_index >= text_size:
                        dist_pred[i][j][0] = 0 #points outside of text
                        continue
                    while np.argmax(arg_pred[i][tgt_index]) != 3: #not first claim token
                        tgt_index -= 1
                        if tgt_index == j: #does not point to claim
                            break
                    dist_pred[i][j][0] = tgt_index - j
                elif src_arg == 3 or (src_arg == 0 and is_claim): #claim
                    if src_arg == 3:
                        is_beginning = True
                    elif src_arg == 0:
                        is_beginning = False

                    is_premise = False
                    is_claim = True
                    if pred_dist == 0:
                        dist_pred[i][j][0] = 0
                        continue
                    tgt_index = j + pred_dist
                    if tgt_index >= text_size:
                        dist_pred[i][j][0] = 0 #points outside of text
                        continue
                    while np.argmax(arg_pred[i][tgt_index]) != 2: #not first premise token
                        tgt_index -= 1
                        if tgt_index == j: #does not point to claim
                            break
                    dist_pred[i][j][0] = tgt_index - j
                f.write(u'phase 1: ' + str(arg_pred[i][j]) + '\t' + str(dist_pred[i][j]) + '\n')

            # f.write(u'i: ' + str(i) + ' - phase 1: ' + str(dist_pred[i]) + '\n')
            k = 0
            while k < text_size: #ensure uniformity: all tokens in src arg comp point to same tgt token
                src_orig = k
                src_arg = np.argmax(arg_pred[i][k])
                pred_dist = dist_pred[i][k][0]
                if src_arg == 1: #non-arg
                    if pred_dist > 0:
                        print('i done goofed')
                        dist_pred[i][k][0] = 0
                    k += 1
                    continue

                if pred_dist == 0:
                    tgt_freq = {'none': 1}
                else:
                    tgt_freq = {pred_dist: 1}

                m =  k + 1
                while (m < text_size) and (np.argmax(arg_pred[i][m]) == 0):
                    pred_dist = dist_pred[i][m][0]
                    if pred_dist == 0:
                        tgt_orig = 'none'
                    else:
                        tgt_orig = pred_dist + (m - src_orig)

                    if tgt_orig in tgt_freq.keys():
                        tgt_freq[tgt_orig] += 1
                    else:
                        tgt_freq[tgt_orig] = 1
                    m += 1
                k = m

                max_value = max(tgt_freq.values()) #get most common decision
                most_freq = []
                for dist in tgt_freq.keys():
                    if tgt_freq[dist] == max_value:
                        most_freq.append(dist)
                if len(most_freq) > 1:
                    # most_freq = [0] #decides none
                    if 'none' in most_freq:
                        most_freq = [0]
                    else:
                        for n in range(0, len(most_freq)):
                            most_freq[n] = int(most_freq[n])
                        most_freq = [min(most_freq)] #decides closest
                if most_freq[0] == 'none' or most_freq[0] == 0:
                    for l in range(src_orig, k):
                        dist_pred[i][l] = [0]
                else:
                    for l in range(src_orig, k):
                        dist_pred[i][l] = most_freq
                        most_freq[0] -= 1

            f.write(u'i: ' + str(i) + ' - phase 2: ' + str(dist_pred[i]) + '\n')
        f.close()
        print('=========== DONE ===========') #debug
        return (arg_pred, dist_pred)

    def replace_argument_tag(self, y_pred_class, unencodedY):
        all_texts_true = []
        all_texts_pred = []
        number_of_texts = len(y_pred_class)
        for i in range(0, number_of_texts):
            true_classes = np.trim_zeros(unencodedY[i])
            text_size = len(true_classes)

            result = np.resize(y_pred_class[i], (text_size, self.num_tags)) #(text size, n tags) -> [[0, 1, 0, 0], [0, 0, 1, 0], ...]
            pred_classes = np.argmax(result, axis=1) #[1, 2, ...] -> size of text
            pred_classes = np.add(pred_classes, 1) #[2, 3, ...] -> to match unencodedY

            # is_premise = False
            # is_claim = False

            # for j in range(0, text_size):
            #     arg = true_classes[j]
            #
            #     if arg == 1:
            #         if is_premise:
            #             true_classes[j] = 3
            #         elif is_claim:
            #             true_classes[j] = 4
            #     elif arg == 2:
            #         is_premise = False
            #         is_claim = False
            #     elif arg == 3:
            #         is_premise = True
            #         is_claim = False
            #     elif arg == 4:
            #         is_premise = False
            #         is_claim = True

            all_texts_true.append(true_classes)

            # for k in range(0, text_size):
            #     arg = pred_classes[k]
            #
            #     if arg == 1:
            #         if is_premise:
            #             pred_classes[k] = 3
            #         elif is_claim:
            #             pred_classes[k] = 4
            #     elif arg == 2:
            #         is_premise = False
            #         is_claim = False
            #     elif arg == 3:
            #         is_premise = True
            #         is_claim = False
            #     elif arg == 4:
            #         is_premise = False
            #         is_claim = True

            all_texts_pred.append(pred_classes)

        return (all_texts_true, all_texts_pred)
