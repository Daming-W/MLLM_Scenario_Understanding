import os
import json

res_json = '/root/LLaVA_SU/su_data/outputs/0524_1346.json'

cnt={
    "smoking": 0,
    "hailing": 0,
    "ped_on_lawn": 0,
    "crowded": 0,
    "fire": 0,
    "trash": 0,
    "illegal_parking": 0
}

with open(res_json,'r') as res:
    data = json.load(res)

    for img, res_dict in data.items():
        pred = res_dict['pred']
        ground_truth = res_dict['ground_truth']

        try:
            indice = [i for i, x in enumerate(ground_truth) if x == 1][0]
            if pred[indice]!=1:
                cnt[list(res_dict["bool_dict"].keys())[indice]] += 1
        except:
            pass
print(cnt)
