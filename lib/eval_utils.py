import os
import json

def bool2binary(bool_dict):
    return [1 if i else 0 for i in bool_dict.values()]

def load_annotations(json_path):
    with open(json_path,'r') as j:
            onehot = json.load(j)
    return onehot

def su_evaluate(json_path):
    images = [i for i in json_path.keys()]
    annotations = [j for j in json_path.values()]
