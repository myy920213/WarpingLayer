import json
import numpy as np

with open('pre_stores/mini_imagenet/label_dict.txt') as f:
    data = f.read()
  
print("Data type before reconstruction : ", type(data))
      
# reconstructing the data as a dictionary
labels = json.loads(data)
num_labels = len(labels)
print('len_label', num_labels)
print("Data type after reconstruction : ", type(labels))
#print(js)

with open('pre_stores/mini_imagenet/imp.txt') as f:
    imp_data = f.readlines()

imp_data = [x.strip() for x in imp_data]
#print(imp_data)
num_imp = len(imp_data)
print('len_imp', num_imp)

imp_matrix = np.zeros((num_labels, num_imp))
for i, item in enumerate(imp_data):
    s1, s2 = item.split()
    l1 = labels[s1]
    l2 = labels[s2]
    imp_matrix[l1][i] = 1
    imp_matrix[l2][i] = -1
print(imp_matrix.shape)
np.save('pre_stores/mini_imagenet/imp_matrix', imp_matrix)
with open('pre_stores/mini_imagenet/exc.txt') as f:
    exc_data = f.readlines()

exc_data = [x.strip() for x in exc_data]
#print(imp_data)
num_exc = len(exc_data)
print('len_exc', num_exc)
exc_matrix = np.zeros((num_labels, num_exc))
for i, item in enumerate(exc_data):
    s1, s2 = item.split()
    l1 = labels[s1]
    l2 = labels[s2]
    exc_matrix[l1][i] = 1
    exc_matrix[l2][i] = 1
print(exc_matrix.shape)
np.save('pre_stores/mini_imagenet/exc_matrix', exc_matrix)





