import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.autoround import AutoRoundModifier

from llmcompressor.utils import dispatch_for_generation

### Code is from https://github.com/vllm-project/llm-compressor/blob/3f934a568e7b3e08f6ff61b1d8e073235aaf071f/examples/quantization_w4a16/llama3_example.py

def apply_quantization(model, tokenizer, dataset_id, dataset_split, num_calibration_samples, max_sequence_length):
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    # Load dataset and preprocess.
    ds = load_dataset(dataset_id, split=f"{dataset_split}[:{num_calibration_samples}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Configure the quantization algorithm to run.
    #   * quantize the weights to 4 bit with GPTQ with a group size 128
    # recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    recipe = GPTQModifier(targets="Linear", scheme="W8A16", ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.experts.*.*_proj$"])

    # recipe = AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"], duo_scaling="both")
    # recipe = AutoRoundModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], iters=200)

    # Apply algorithms.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
        # shuffle_calibration_samples=False,
    )

if __name__ == '__main__':
    # Select model and load it.
    model_path = "OLMoE-1B-7B-0924-Instruct/"

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Select calibration dataset.
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Select number of samples. 512 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 256
    MAX_SEQUENCE_LENGTH = 2048

    apply_quantization(model, tokenizer, DATASET_ID, DATASET_SPLIT, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH)

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save to disk compressed.
    SAVE_DIR = model_path.rstrip("/").split("/")[-1] + "-W8A16-GPTQ"
    print(SAVE_DIR)
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
