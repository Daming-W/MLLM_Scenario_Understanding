import os
import json
import numpy as np
from sklearn.metrics import recall_score,precision_score,accuracy_score

import numpy as np

def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = np.sum(np.logical_and(y_true[i], y_pred[i]))
        q = np.sum(np.logical_or(y_true[i], y_pred[i]))
        if q != 0:
            count += p / q
        else:
            pass
    return count / y_true.shape[0]

def Precision(y_true, y_pred):
    count = 0
    valid_samples = 0
    for i in range(y_true.shape[0]):
        pred_sum = np.sum(y_pred[i])
        if pred_sum == 0:
            continue  # 跳过没有任何预测的样本
        count += np.sum(np.logical_and(y_true[i], y_pred[i])) / pred_sum
        valid_samples += 1
    return count / valid_samples if valid_samples > 0 else 0

def MacroRecall(y_true, y_pred):
    count = 0
    valid_samples = 0
    for i in range(y_true.shape[0]):
        true_sum = np.sum(y_true[i])
        if true_sum == 0:
            continue
        count += np.sum(np.logical_and(y_true[i], y_pred[i])) / true_sum
        valid_samples += 1
    return count / valid_samples if valid_samples > 0 else 0

def MicroRecall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    all_positives = np.sum(y_true)
    return true_positives / all_positives if all_positives > 0 else 0

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

def eval_from_json(json_file):

    with open(json_file,'r') as j:

        data = json.load(j)
        all_pred,all_gt = [],[]

        for img, res_dict in data.items():
            pred = res_dict['pred']
            ground_truth = res_dict['ground_truth']

            all_pred.append(pred)
            all_gt.append(ground_truth)

        print('recall macro',recall_score(np.array(all_gt),np.array(all_pred),average='macro'))
        print('recall micro',recall_score(np.array(all_gt),np.array(all_pred),average='micro'))
    
        print('precision macro',precision_score(np.array(all_gt),np.array(all_pred),average='macro'))
        print('precision micro',precision_score(np.array(all_gt),np.array(all_pred),average='micro'))

        print('acc',accuracy_score(np.array(all_gt),np.array(all_pred)))


if __name__=='__main__':

    eval_from_json(None)