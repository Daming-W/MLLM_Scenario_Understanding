import os 
import json
from glob import glob

with open('/root/LLaVA_SU/images_su/annotations.json','w') as j:
    info={}
    for img in glob('/root/LLaVA_SU/images_su/eval/*.jpg'):
        info[img]=None
    json.dump(info,j,indent=4)