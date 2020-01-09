import networkx as nx
import matplotlib.pyplot as plt
import os

golden_path = 'Input_G/Pt'
predict_path = 'Output_G/Pt'
# golden_path = 'Input_G/En'
# predict_path = 'Output_G/En'

def node_subst_cost(n1, n2):
    text1 = n1['label'].lower()
    text2 = n2['label'].lower()
    text1 = text1.replace('\n', '')
    text2 = text2.replace('\n', '')
    text1 = text1.split(' ')
    text2 = text2.split(' ')

    size1 = len(text1)
    size2 = len(text2)
    matching = False
    start = 0
    spans = {}
    max_len = 0
    max_i = 0
    i = j = 0

    while i < size1:
        while j < size2 and text1[i] != text2[j]:
            j += 1
        if j < size2:
            start = i
            while i < size1 and j < size2 and text1[i] == text2[j]:
                i += 1
                j += 1
            spans[start] = i
            if i - start > max_len:
                max_len = i - start
                max_i = start
        else:
            j = 0
            i += 1
    cost = max(size1, size2) - max_len
    return cost


golden_files = [f for f in os.listdir(golden_path) if os.path.isfile(os.path.join(golden_path, f))]

total_ged = []
data = {}
for filename in golden_files:
    print(filename)
    golden_graph = nx.drawing.nx_pydot.read_dot(golden_path + '/' + filename)
    predict_graph = nx.drawing.nx_pydot.read_dot(predict_path + '/' + filename)

    # nx.drawing.nx_pylab.draw_networkx(golden_graph)
    # plt.pause(0.001)
    # input("Press [enter] to continue.")

    ged = nx.algorithms.similarity.graph_edit_distance(golden_graph, predict_graph, node_subst_cost=node_subst_cost)
    total_ged.append(ged)
    if ged in data.keys():
        data[ged] += 1
    else:
        data[ged] = 1

print('Avg GED:', sum(total_ged)/len(golden_files))
print('Min GED:', min(total_ged), '\tMax GED:', max(total_ged))
x = []
y = []
for key in data.keys():
    x.append(key)
    y.append(data[key])
plt.bar(x, y)
plt.pause(0.001)
input("Press [enter] to continue.")
