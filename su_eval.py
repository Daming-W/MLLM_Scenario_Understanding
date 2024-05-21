import os
import sys
import json
import time
import argparse

import torch

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from lib.mapper import Mapper
from lib.predictor import su_inference
from lib.eval_utils import *

from sklearn.metrics import recall_score, accuracy_score
import numpy as np
from tqdm import tqdm

if __name__=='__main__':

    import warnings
    warnings.filterwarnings("ignore")

    # input arg
    parser = argparse.ArgumentParser(description='Run LLaVA model inference.')
    parser.add_argument('--json_file', type=str, required=True, help='Path to the image file')
    input_args = parser.parse_args()
    
    # load images
    with open(input_args.json_file,'r') as j:
        data_dict=json.load(j)
    
    # define input args
    args = type('Args', (), {
        "model_path": "liuhaotian/llava-v1.6-vicuna-7b",
        "model_base": None,
        "model_name": get_model_name_from_path("liuhaotian/llava-v1.6-vicuna-7b"),
        "query": None,
        "conv_mode": None,
        "image_file": None,
        "sep": ",",
        "temperature": 0.1,
        "top_p": 0.99,
        "num_beams": 1,
        "max_new_tokens": 32
    })()

    # load pre-trained model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)
    print(f'\n ** finish loading model {model_name} ** \n')
    
    # load query prompt
    with open('/root/LLaVA_SU/su_prompts/prompt.json','r') as f:
        query_dict=json.load(f)
    print(f'query_dict:\n{query_dict}\n')

    recall,accuracy,latency=[],[],[]

    for sample, gt in tqdm(data_dict.items()):

        args.image_file=sample

        # inference with returning a dict of answer text
        start_time = time.time()
        answer_dict=su_inference(args, model_name, tokenizer, model, image_processor, None, query_dict)
        end_time = time.time()
        execution_time = end_time - start_time
        latency.append(execution_time)
        # print(f"\n{sample} --> su_inference with: {execution_time:.2f} sec\n")

        # inference with returning a dict of boolean
        mapper=Mapper(answer_dict)
        bool_dict=mapper.answer2bool()

        # convert bool dict to onehot
        pred=bool2binary(bool_dict)
        print(pred)

        # append result
        recall.append(recall_score(gt, pred))
        accuracy.append(accuracy_score(gt, pred))
    
    print(recall,'\n',accuracy,'\n',latency,'\n')
    print(np.mean(recall),np.mean(accuracy),np.mean(latency))

        