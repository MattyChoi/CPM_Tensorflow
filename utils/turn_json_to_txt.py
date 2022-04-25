import os
import json
import sys

path = r'C:\Users\matth\OneDrive\Documents\Storage\CSCI_5561\cpm-tf\utils'
data_dir = os.path.join(path, 'dataset')

with open(os.path.join(data_dir, 'train_annotation.json')) as f:
    dic = json.load(f)
    
    g = open(os.path.join(data_dir, 'train', 'labels.txt'), mode = "w")
    for data in dic['data']:
        file = data['file']
        bbox = data['bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        landmarks = data['landmarks']
        contents = file + ' ' + ' '.join(map(str, bbox)) + ' ' + ' '.join(map(str, landmarks))
        g.write(contents + '\n')
    g.close()

# with open(os.path.join(data_dir, 'val_annotation.json')) as f:
#     dic = json.load(f)
    
#     g = open(os.path.join(data_dir, 'val', 'labels.txt'), mode = "w")
#     for data in dic['data']:
#         file = data['file']
#         bbox = data['bbox']
#         bbox[2] += bbox[0]
#         bbox[3] += bbox[1]
#         landmarks = data['landmarks']
#         contents = file + ' ' + ' '.join(map(str, bbox)) + ' ' + ' '.join(map(str, landmarks))
#         g.write(contents + '\n')
#     g.close()