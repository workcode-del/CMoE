import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from CMoE_utils import *
from CMoE_model import *
from zero_eval import *
from sft_utils import simple_sft
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompress import apply_quantization

def get_llama(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    # model.seqlen = 4096
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def get_olmoe(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import OlmoeForCausalLM

    # model = OlmoeForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')

    model.seqlen = 2048
    return model

def get_deepseek_v2_lite_gptq(model_path):
    from auto_gptq import AutoGPTQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # model = AutoGPTQForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    #     device_map="auto",
    #     trust_remote_code=True,
    #     quantize_config=None,
    #     use_safetensors=True
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True
    )

    model.seqlen = 2048

    return model, tokenizer

def get_auto(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True
    )

    model.seqlen = 2048

    return model, tokenizer

def load_model(model_path):
    if 'llava' in model_path.lower():
        model = get_llava(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'olmoe' in model_path.lower():
        model = get_olmoe(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'deepseek-v2-lite' in model_path.lower():
        model, tokenizer = get_deepseek_v2_lite_gptq(model_path)
    elif 'llama' in model_path.lower():
        model = get_llama(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model, tokenizer = get_auto(model_path)
    model.eval()
    return model, tokenizer

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    args = parser.parse_args()
    
    print("Loading model: ", args.model.lower())
    model, tokenizer = load_model(args.model)

    print("model: ", args.model)

    ppl = []
    datasets = ['wikitext2', 'c4-new']
    # datasets = ['wikitext2', ]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = 1
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        ppl.append(f"{dataset}: {ppl_i}")
        print("PPL on {}: {:.4f}".format(dataset, ppl_i))
