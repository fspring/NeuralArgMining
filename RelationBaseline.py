import numpy as np
## for each premise find closest claim ahead
def predict_baseline_distances_claim(self, pred_tags):
    pred_dists = []
    for i in range(0, len(pred_tags)):
        is_premise = False
        is_claim = False
        file_dists = []
        text_size = len(pred_tags[i])
        for j in range(0, text_size):
            arg_class = np.argmax(pred_tags[i][j])
            dist = 0
            if arg_class == 1: # is O
                is_premise = False
                is_claim = False
            elif arg_class == 2: # is P
                is_premise = True
                is_claim = False
                for k in range(j+1, text_size): # look ahead
                    arg_rel = np.argmax(pred_tags[i][k])
                    dist += 1
                    if arg_rel == 3: # is C
                        break
            elif arg_class == 3: # is C
                is_premise = False
                is_claim = True
            elif arg_class == 0 and is_premise:
                if dist > 0:
                    dist -= 1
            elif arg_class == 0 and not is_claim and not is_premise:
                pred_tags[i][j][0] = 0
                pred_tags[i][j][1] = 1

            file_dists.append([dist])
        pred_dists.append(file_dists)
    return pred_dists

## for each premise find closest arg_component ahead
def predict_baseline_distances_next(self, pred_tags):
    pred_dists = []
    for i in range(0, len(pred_tags)):
        file_dists = []
        text_size = len(pred_tags[i])
        for j in range(0, text_size):
            arg_class = np.argmax(pred_tags[i][j])
            dist = 0
            if arg_class == 2:
                k = j+1
                while k < text_size and np.argmax(pred_tags[i][k]) == 0:
                    dist += 1
                    k += 1
                for n in range(k, text_size):
                    arg_rel = np.argmax(pred_tags[i][n])
                    if arg_rel == 1:
                        dist += 1
                    elif arg_rel == 3:
                        dist += 1
                        break
            file_dists.append([dist])
        pred_dists.append(file_dists)
    return pred_dists

## for each premise find closest claim ahead and for each claim find closest premise ahead
def predict_baseline_distances_all_next(self, pred_tags):
    pred_dists = []
    for i in range(0, len(pred_tags)):
        file_dists = []
        text_size = len(pred_tags[i])
        is_arg = False
        for j in range(0, text_size):
            arg_class = np.argmax(pred_tags[i][j])
            dist = 0
            if arg_class == 1: #is O
                is_arg = False
            elif arg_class == 0 and not is_arg: #illegal I tag
                #change tag I to O
                pred_tags[i][j][0] = 0
                pred_tags[i][j][1] = 1
            elif arg_class == 2: #is P
                is_arg = True
                k = j+1
                while k < text_size and np.argmax(pred_tags[i][k]) == 0:
                    dist += 1
                    k += 1
                for n in range(k, text_size):
                    arg_rel = np.argmax(pred_tags[i][n])
                    if arg_rel == 1:
                        dist += 1
                    elif arg_rel == 3:
                        dist += 1
                        break
            elif arg_class == 3: #is C
                is_arg = True
                k = j+1
                while k < text_size and np.argmax(pred_tags[i][k]) == 0:
                    dist += 1
                    k += 1
                for n in range(k, text_size):
                    arg_rel = np.argmax(pred_tags[i][n])
                    if arg_rel == 1:
                        dist += 1
                    elif arg_rel == 2:
                        dist += 1
                        break
            file_dists.append([dist])
        pred_dists.append(file_dists)
    return (pred_tags, pred_dists)
