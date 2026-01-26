#!/bin/sh

# MODEL_PATH=""
# python run_cmoe.py $MODEL_PATH wikitext2 --new-eval --nshared 2 --nactivated 2 --nexperts 16 --nsamples 64 --extra-lr 0.001 --bias-speed 0.001 --sft-bsz 1 --carve-bsz 1 --epoch 0
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 

python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m4422
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m4332
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m4222
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m4221
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m3333
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m3322
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m3222
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a4s0m3221

python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m4422
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m4332
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m4222
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m4221
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m3333
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m3322
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m3222
python run_cmoe.py ~/models/Llama-2-7b-hf/ wikitext2 --new-eval --nshared 0 --nactivated 4  --nexperts 4  --nsamples 64 --quant-scheme a3s0m3221

# python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 8  --nexperts 64  --nsamples 64  --quant-scheme a2s0m2
# python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 16  --nexperts 128  --nsamples 64 --quant-scheme a8s0m44
# python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 16  --nexperts 128  --nsamples 64 --quant-scheme a8s0m42
# python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 32  --nexperts 256  --nsamples 64 --quant-scheme a8s0m4422

# python run_cmoe.py ~/models/deepseek-moe-16b-base/ wikitext2 --new-eval --nshared 0 --nactivated 32 --nexperts 256 --nsamples 64 --quant-scheme a8s2m3222
# python run_cmoe.py ~/models/deepseek-moe-16b-base/ wikitext2 --new-eval --nshared 0 --nactivated 16 --nexperts 128 --nsamples 64 --quant-scheme a8s2m22
# python run_cmoe.py ~/models/deepseek-moe-16b-base/ wikitext2 --new-eval --nshared 0 --nactivated 8  --nexperts 64  --nsamples 64 --quant-scheme a8s2m2

# python run_cmoe.py ~/models/DeepSeek-V2-Lite/ wikitext2 --new-eval --nshared 0 --nactivated 32 --nexperts 256 --nsamples 64 --quant-scheme a8s2m3221
python run_cmoe.py ~/models/DeepSeek-V2-Lite/ wikitext2 --new-eval --nshared 0 --nactivated 32 --nexperts 256 --nsamples 64 --quant-scheme a8s4m3221
python run_cmoe.py ~/models/DeepSeek-V2-Lite/ wikitext2 --new-eval --nshared 0 --nactivated 16 --nexperts 128 --nsamples 64 --quant-scheme a8s4m42
# python run_cmoe.py ~/models/DeepSeek-V2-Lite/ wikitext2 --new-eval --nshared 0 --nactivated 8  --nexperts 64  --nsamples 64 --quant-scheme a8s2m2

