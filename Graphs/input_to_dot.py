import os

# dir = 'CorpusOutputPunctuation/rel'
dir = 'essaysClaimsPremisesPunctuation/rel/both'

# graph_dir = 'Graphs/Input_G/Pt'
graph_dir = 'Graphs/Input_G/En'

files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

for filename in files:
    fd = open(dir + '/' + filename, 'r', encoding='utf-8')
    entries = fd.readlines()
    fd.close()
    fields = entries[0].split('\t') #[text, (IO,arg,dist)]

    tag = fields[1].split(',')
    arg = tag[1]
    text = fields[0]
    if tag[-1][:-2] == '|':
        dist = 0
    else:
        dist = int(tag[-1][:-2])
    begin = end = 0

    spans = {} #{id: [arg, (begin, end), link, text]}

    i = 1
    while i < len(entries):
        fields = entries[i].split('\t') #[text, (IO,arg,dist)]
        tag = fields[-1].split(',')
        while tag[1] == arg:
            end = i
            text += ' ' + fields[0]
            i += 1
            if i >= len(entries):
                break
            fields = entries[i].split('\t') #[text, (IO,arg,dist)]
            tag = fields[-1].split(',')
        spans[begin] = [arg, (begin, end), begin + dist, text]
        if i >= len(entries):
            break
        tag = fields[-1].split(',')
        arg = tag[1]
        text = fields[0]
        if tag[-1][:-2] == '|':
            dist = 0
        else:
            dist = int(tag[-1][:-2])
        begin = end = int(i)
        i += 1

    graph_header = 'digraph G {\n'
    edges = ''
    attrs = ''
    for span_id in spans.keys():
        if spans[span_id][0] == 'premise':
            link = spans[span_id][2]
            if link == span_id:
                continue
            edges += '\t' + str(span_id) + ' -> ' + str(link) + ';\n'
            attrs += '\t' + str(span_id) + ' [label=\"' + spans[span_id][3] + '\",color=blue];\n'
            attrs += '\t' + str(link) + ' [label=\"' + spans[link][3] + '\",color=green];\n'

        elif spans[span_id][0] == 'claim':
            link = spans[span_id][2]
            if link == span_id:
                continue
            edges += '\t' + str(link) + ' -> ' + str(span_id) + ';\n'
            attrs += '\t' + str(span_id) + ' [label=\"' + spans[span_id][3] + '\",color=green];\n'
            attrs += '\t' + str(link) + ' [label=\"' + spans[link][3] + '\",color=blue];\n'

    fd = open(graph_dir + '/' + filename, 'w', encoding='utf-8')
    fd.write(graph_header + edges + attrs + '}')
    fd.close()
