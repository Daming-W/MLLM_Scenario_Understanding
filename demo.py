from builtins import print
import os
import sys
import json
import time

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from lib.mapper import mapper
from lib.predictor import su_inference


if __name__=='__main__':

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    #input image
    image_file = "/root/LLaVA_SU/images_su/lsu/smoke1.jpg"

    # check image 
    if not os.path.exists(image_file):
        print(f"Error: Image file {image_file} does not exist.")
        sys.exit(1) 

    # define input args
    args = type('Args', (), {
        "model_path": "liuhaotian/llava-v1.5-7b",
        "model_base": None,
        "model_name": get_model_name_from_path("liuhaotian/llava-v1.5-7b"),
        # liuhaotian/llava-v1.5-7b
        # liuhaotian/llava-v1.5-13b
        "query": None,
        "conv_mode": None,
        "image_file": None,
        "sep": ",",
        "temperature": 0.2,
        "top_p": 0.99,
        "num_beams": 1,
        "max_new_tokens": 16
    })()

    # check image 
    if not os.path.exists(args.image_file):
        print(f"Error: Image file {args.image_file} does not exist.")
        sys.exit(1) 
    
    # load pre-trained model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)
    print(f'\n ** finish loading model {model_name} ** \n')

    # load query prompt
    with open('/root/LLaVA_SU/lib/prompt.json','r') as f:
        query_dict=json.load(f)
    print(f'query_dict:\n{query_dict}\n')
    
    # inference with returning a dict of answer text
    start_time = time.time()
    answer_dict=su_inference(args, model_name, tokenizer, model, image_processor, args.image_file, query_dict)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nsu_inference within: {execution_time:.2f} sec")

    print(f'answer_dict:\n{answer_dict}\n')
    # print(logits)

    # inference with returning a dict of boolean
    mapper=mapper(answer_dict)
    bool_dict=mapper.answer2bool()
    print(f'bool_dict:\n{bool_dict}\n')