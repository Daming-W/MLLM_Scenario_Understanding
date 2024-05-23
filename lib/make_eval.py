import os
import json
from glob import glob

label_dict = {
    "smoking": [1, 0, 0, 0, 0, 0, 0],
    "hailing": [0, 1, 0, 0, 0, 0, 0],
    "ped_on_lawn": [0, 0, 1, 0, 0, 0, 0],
    "crowded": [0, 0, 0, 1, 0, 0, 0],
    "fire": [0, 0, 0, 0, 1, 0, 0],
    "trash": [0, 0, 0, 0, 0, 1, 0],
    "illegal_parking": [0, 0, 0, 0, 0, 0, 1]
}

with open('/root/LLaVA_SU/su_data/annotations10.json', 'w') as file:
    info = {}
    for img in glob('/root/LLaVA_SU/su_data/eval_10/*.jpg'):
        info[img] = [0, 0, 0, 0, 0, 0, 0]
        for label, values in label_dict.items():
            if label in img:
                info[img] = values
    print(info)
    json.dump(info, file, indent=4)
