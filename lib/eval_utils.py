import os
import json

from sklearn.metrics import recall_score

def su_recall_score(y_true, y_pred):
    if all(y == 0 for y in y_true) and all(y == 0 for y in y_pred):
        return 1.0
    else:
        return recall_score(y_true, y_pred, average='binary')

def bool2binary(bool_dict):
    return [1 if i else 0 for i in bool_dict.values()]

def load_annotations(json_path):
    with open(json_path,'r') as j:
            onehot = json.load(j)
    return onehot

def su_evaluate(json_path):
    images = [i for i in json_path.keys()]
    annotations = [j for j in json_path.values()]
