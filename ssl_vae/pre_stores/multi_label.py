import json
import torch

with open('cifar_labels.txt') as f:
    data = f.read()
    labels = json.loads(data)

with open('cifar_fine_labels.txt') as f:
    data = f.read()
    fine_labels = json.loads(data)

with open('imp.txt') as f:
    imp_data = f.readlines()

imp_data = [x.strip() for x in imp_data]
#print(imp_data)
num_imp = len(imp_data)
print('len_imp', num_imp)

label_imp = {}
for i, (k, v) in enumerate(labels.items()):
    current_label = k
    parents = [labels[current_label]]
    print('i', i)
    while current_label != "entity":
        print('current:', current_label)
        for pairs in imp_data:
            s1, s2 = pairs.split()
            if current_label == s1:
                current_label = s2
                parents.append(fine_labels[current_label])
    label_imp[v] = parents

with open('label_imp.txt', 'w') as convert_file:
    convert_file.write(json.dumps(label_imp))
'''
l = torch.tensor([[1,3,4],[0,3], [0,3,5], [1,4], [0,2,3]])
oh = torch.zeros(5,6)
print(oh.scatter_(1, l, 1))
'''