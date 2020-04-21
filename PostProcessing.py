import numpy as np

class PostProcessing:
    def __init__(self, num_tags, tags):
        self.num_tags = num_tags
        self.tags = tags

    def correct_dist_prediction(self, arg_pred, dist_pred, unencodedY):
        print('=========== CORRECTING ===========') #debug
        f = open('correction_debug.txt', 'w')
        for i in range(0, len(dist_pred)):
        # for i in range(16, 17):
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
                    while np.argmax(arg_pred[i][tgt_index]) != 3 or np.argmax(arg_pred[i][tgt_index]) != 2: #not first claim token or premise token
                        tgt_index -= 1
                        if tgt_index == j: #does not point to claim nor premise
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
                        if tgt_index == j: #does not point to premise
                            break
                    dist_pred[i][j][0] = tgt_index - j
                f.write(u'phase 1: ' + str(arg_pred[i][j]) + '\t' + str(dist_pred[i][j]) + '\n')

            # f.write(u'i: ' + str(i) + ' - phase 1: ' + str(dist_pred[i]) + '\n')
            k = 0
            while k < text_size: #ensure uniformity: all tokens in src arg comp point to same tgt token
                # print('k\t', k)
                src_orig = k
                src_arg = np.argmax(arg_pred[i][k])
                pred_dist = dist_pred[i][k][0]
                # print('src arg\t', src_arg, '\tdist\t', pred_dist)
                if src_arg == 1: #non-arg
                    if pred_dist > 0:
                        print('error in correction dist')
                        dist_pred[i][k][0] = 0
                    k += 1
                    continue
                # print('src arg is not 1')
                if pred_dist == 0:
                    tgt_freq = {'none': 1}
                else:
                    tgt_freq = {pred_dist: 1}
                # print('dist\t', pred_dist, '\ttgt freq dict init\t', tgt_freq)
                m =  k + 1
                while (m < text_size) and (np.argmax(arg_pred[i][m]) == 0):
                    # print('m\t', m)
                    pred_dist = dist_pred[i][m][0]
                    # print('dist\t', pred_dist)
                    if pred_dist == 0:
                        tgt_orig = 'none'
                    else:
                        tgt_orig = pred_dist + (m - src_orig)

                    if tgt_orig in tgt_freq.keys():
                        tgt_freq[tgt_orig] += 1
                    else:
                        tgt_freq[tgt_orig] = 1
                    # print('updated tgt freq dict\t', tgt_freq)
                    m += 1
                k = m
                # print('k\t', k)

                # print('finding most common tgt')

                max_value = max(tgt_freq.values()) #get most common decision
                most_freq_list = []
                most_freq = 0
                for dist in tgt_freq.keys():
                    if tgt_freq[dist] == max_value:
                        most_freq_list.append(dist)
                if len(most_freq_list) > 1:
                    # most_freq_list = [0] #decides none
                    if 'none' in most_freq_list:
                        most_freq = 0
                    else:
                        for n in range(0, len(most_freq_list)):
                            most_freq_list[n] = int(most_freq_list[n])
                        most_freq = min(most_freq_list) #decides closest
                elif len(most_freq_list) == 1:
                    most_freq = most_freq_list[0]

                # print('most common tgt\t', most_freq)

                if most_freq == 'none' or most_freq == 0:
                    # print('most freq is 0 or none')
                    # print('correcting from src orig to k:\t', src_orig, 'to', k)
                    for l in range(src_orig, k):
                        dist_pred[i][l] = [0]
                else:
                    # print('most freq is some value')
                    # print('correcting from src orig to k:\t', src_orig, 'to', k)
                    for l in range(src_orig, k):
                        dist_pred[i][l] = [most_freq]
                        most_freq -= 1
                        # print('l', l)
                        # print('most freq', most_freq)
                        # print('dists 100-107', dist_pred[i][100], dist_pred[i][101], dist_pred[i][102], dist_pred[i][103], dist_pred[i][104], dist_pred[i][105], dist_pred[i][106], dist_pred[i][107])

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
