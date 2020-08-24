import os

dir = 'allRelationTags'

files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

pt_prems = 0
pt_claims = 0
en_prems = 0
en_claims = 0
for filename in files:
    # print(filename)
    # if 'essay' in filename:
    #     continue
    fd = open(dir + '/' + filename, 'r', encoding='utf-8')
    text = fd.read()
    fd.close()
    # fd = open('allTagsPunctuation/' + filename, 'r', encoding='utf-8')
    # text_2 = fd.readlines()
    # fd.close()
    # assert len(text) == len(text_2)
    # start = -1
    # end = -1
    # for i in range(len(text_2)):
    #     if i == 0 and text_2[i][3] == 'c':
    #         start = i
    #         while i < len(text_2)-1 and text_2[i+1][3] == 'c':
    #             i += 1
    #             end = i
    #         for j in range(start, end+1):
    #             print(j)
    #             assert text[j][1] == 'C' or text[j][1] == 'I'
    #     elif text_2[i][3] == 'c' and text_2[i-1][3] != 'c':
    #         start = i
    #         while i < len(text_2)-1 and text_2[i+1][3] == 'c':
    #             i += 1
    #             end = i
    #         for j in range(start, end+1):
    #             print(j)
    #             assert text[j][1] == 'C' or text[j][1] == 'I'



    if 'essay' in filename:
        en_claims += text.count('C')
        en_prems += text.count('P')
    else:
        pt_claims += text.count('C')
        pt_prems += text.count('P')

print('Span Count')
print('English:\tclaims - '+str(en_claims)+'\tpremises - '+str(en_prems))
print('Portuguese:\tclaims - '+str(pt_claims)+'\tpremises - '+str(pt_prems))
