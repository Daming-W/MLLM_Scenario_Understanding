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


if __name__=='__main__':

    import warnings
    warnings.filterwarnings("ignore")

    #input image
    parser = argparse.ArgumentParser(description='Run LLaVA model inference.')
    parser.add_argument('--image_file', type=str, required=True, help='Path to the image file')
    input_args = parser.parse_args()

    # check image 
    if not os.path.exists(input_args.image_file):
        print(f"Error: Image file {input_args.image_file} does not exist.")
        sys.exit(1) 

    # define input args
    args = type('Args', (), {
        "model_path": "liuhaotian/llava-v1.6-vicuna-7b",
        "model_base": None,
        "model_name": get_model_name_from_path("liuhaotian/llava-v1.6-vicuna-7b"),
        "query": None,
        "conv_mode": None,
        "image_file": input_args.image_file,
        "sep": ",",
        "temperature": 0.2,
        "top_p": 0.7,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    # pre-trained options
    '''
    liuhaotian/llava-v1.5-7b
    liuhaotian/llava-v1.5-13b
    liuhaotian/llava-v1.6-7b        
    liuhaotian/llava-v1.6-34b
    '''

    # load pre-trained model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)
    print(f'\n ** finish loading model {model_name} ** \n')
    
    # load query prompt
    with open('/root/LLaVA_SU/su_prompts/cotprompt.json','r') as f:
        query_dict=json.load(f)
    print(f'query_dict:\n{query_dict}\n')
    
    # inference with returning a dict of answer text
    start_time = time.time()
    answer_dict=su_inference(args, model_name, tokenizer, model, image_processor, None, query_dict)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nsu_inference with: {execution_time:.2f} sec\n")

    print(f'answer_dict:\n{answer_dict}\n')
    # print(logits)

    # inference with returning a dict of boolean
    mapper=Mapper(answer_dict)
    bool_dict=mapper.answer2bool()
    print(f'bool_dict:\n{bool_dict}\n')


# title="LLaVA-Scenario-Understanding",
# description="Choose a preset image or upload your own image to classify issues --> \
#     Smoking / Pedestrian step on lawn / On Fire / Trash or Fallen Leaves / Crowded / Illegal parking / Hailing"
