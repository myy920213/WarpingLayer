'''
import numpy as np
import scipy.io

with open('ab_tensor.npy', 'rb') as f:
    ab = np.load(f)

scipy.io.savemat('ab_tensor.mat', {'ab': ab})
'''

import json
import numpy as np
with open('pre_store/all_labels.txt') as f:
    data = f.read()
  
print("Data type before reconstruction : ", type(data))
      
# reconstructing the data as a dictionary
labels = json.loads(data)
print('after:', type(labels))

with open('cifar_labels.json', 'w') as fp:
    json.dump(labels, fp)