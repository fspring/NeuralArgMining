import os

# graph_dir = 'Input_G/Pt'
# graph_dir = 'Input_G/En'
# graph_dir = 'Output_G/Pt'
graph_dir = 'Output_G/En'

# json_dir = 'Input_G/Json/Pt'
# json_dir = 'Input_G/Json/En'
# json_dir = 'Output_G/Json/Pt'
json_dir = 'Output_G/Json/En'

files = [f for f in os.listdir(graph_dir) if os.path.isfile(os.path.join(graph_dir, f))]

for filename in files:
    fd = open(graph_dir + '/' + filename, 'r', encoding='utf-8')
    entries = fd.readlines()
    fd.close()
    entries = entries[1:-1]
    nodes = set()
    edges = []
    for entry in entries:
        if '->' in entry:
            entry = entry[1:-2]
            edge = entry.split(' ')
            edge.remove('->')
            edges.append(edge)
            nodes.add(edge[0])
            nodes.add(edge[1])
    json_header = '{\"graph\":{\n\"nodes\":[\n'
    json_sec_header = '],\n\"links\":[\n'
    json_end = ']}}'

    json_nodes = ''
    json_edges = ''

    for node in nodes:
        json_nodes += '{\"id\": \"' + node + '\"},\n'
    json_nodes = json_nodes[:-2] + '\n'

    for edge in edges:
        json_edges += '{\"source\": \"' + edge[0] + '\", \"target\": \"' + edge[1] + '\"},\n'
    json_edges = json_edges[:-2] + '\n'

    fd = open(json_dir + '/' + filename, 'w', encoding='utf-8')
    fd.write(json_header + json_nodes + json_sec_header + json_edges + json_end)
    fd.close()
