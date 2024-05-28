import gradio as gr
from PIL import Image

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
from lib.predictor import su_inference,su_inference_gr

import warnings
warnings.filterwarnings("ignore")

# 定义Args类
class Args:
    def __init__(self, image_file):
        self.model_path = "liuhaotian/llava-v1.6-vicuna-7b"
        self.model_base = None
        self.model_name = get_model_name_from_path(self.model_path)
        self.query = None
        self.conv_mode = None
        self.image_file = image_file
        self.sep = ","
        self.temperature = 0.1
        self.top_p = 0.99
        self.num_beams = 1
        self.max_new_tokens = 32


model_name = get_model_name_from_path("liuhaotian/llava-v1.6-vicuna-7b")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    "liuhaotian/llava-v1.6-vicuna-7b", None, model_name)
print(f'\n ** finish loading model {model_name} ** \n')


with open('/root/LLaVA_SU/su_prompts/prompt.json', 'r') as f:
    query_dict = json.load(f)
print(f'query_dict:\n{query_dict}\n')


def inference(image):
    args = Args(image)
    answer_dict = su_inference_gr(args, model_name, tokenizer, model, image_processor, image, query_dict)
    mapper = Mapper(answer_dict)
    bool_dict = mapper.answer2bool()
    return bool_dict

examples = [
    ["/root/LLaVA_SU/su_data/eval_10/illegal_parking_1.jpg"],
    ["/root/LLaVA_SU/su_data/eval_10/ped_on_lawn_10.jpg"],
    ["/root/LLaVA_SU/su_data/eval_10/smoking_1.jpg"],
    ["/root/LLaVA_SU/su_data/eval_10/crowded_2.jpg"],
    ["/root/LLaVA_SU/su_data/eval_10/ped_on_lawn_6.jpg"],
    ["/root/LLaVA_SU/su_data/eval_10/trash_5.jpg"],
    ["/root/LLaVA_SU/su_data/eval_10/fire_1.jpg"]
]

interface = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="Multimodal-Large-Language-Model-Scenario-Understanding-Demo",
    description="Choose a preset image or upload your own image to classify issues --> \
     Smoking / Pedestrian step on lawn / On Fire / Trash or Fallen Leaves / Crowded / Illegal parking / Hailing",
    examples=examples
)


if __name__ == "__main__":
    interface.launch(share=True)
