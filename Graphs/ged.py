import os
import json
import networkx as nx
import gmatch4py as gm

input_dir = 'Input_G/Json/Pt'
output_dir = 'Output_G/Json/Pt'
fd = open('GED/pt_results.txt', 'w', encoding='utf-8')

# input_dir = 'Input_G/Json/En'
# output_dir = 'Output_G/Json/En'
# fd = open('GED/en_results.txt', 'w', encoding='utf-8')


files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

for filename in files:
    fd_in = open(input_dir + '/' + filename, 'r', encoding='utf-8')
    in_entries = json.load(fd_in)
    fd_in.close()

    fd_out = open(output_dir + '/' + filename, 'r', encoding='utf-8')
    out_entries = json.load(fd_out)
    fd_out.close()

    g_in = nx.readwrite.json_graph.node_link_graph(in_entries['graph'])
    g_out = nx.readwrite.json_graph.node_link_graph(out_entries['graph'])

    ged = gm.GraphEditDistance(1,1,1,1) # all edit costs are equal to 1
    result = ged.compare([g_out,g_in],None)
    norm_result = ged.similarity(result)

    fd.write(u'' + filename + '\n' + str(norm_result) + '\n=================\n')

fd.close()
