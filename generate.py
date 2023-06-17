# Adapted from https://github.com/tloen/alpaca-lora/blob/main/generate.py

base_model = "eachadea/vicuna-7b-1.1"
lora_weights = "models/v2j-vicgen-lora"
load_8bit = False
temperature = 0.1
top_p = 0.75
top_k = 40
num_beams = 4
max_new_tokens = 512
SEED = 42

import argparse
import sys
import torch
import json
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def load_model(base_model, lora_weights, load_8bit):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer

def create_prompt(question, choices=None):
    if choices:
        choices_str = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(choices))
        return f"Below is a question along with the available answer choices. Write a response that succinctly answers the question by selecting the correct choice or choices. Provide a reasoning.\n\n### Question:\n{question}\n\n### Choices:\n{choices_str}\n\n### Response:\n"
    return f"Below is a question. Write a response that succinctly answers the question and provides reasoning.\n\n### Question:\n{question}\n\n### Response:\n"

def generate_response(model, tokenizer, prompt, generation_config, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=False,
            output_scores=False,
            max_new_tokens=max_new_tokens,
        )
    
    generation_output_cpu = generation_output[0].cpu().numpy()
    decoded_sequence = tokenizer.decode(generation_output_cpu, skip_special_tokens=True)
    response_only = decoded_sequence.split("### Response:")
    response_only = response_only[1].strip() if len(response_only)>1 else ""
    return response_only

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_path", 
        type=str, 
        default="prompts.json",
        help="Path to the given prompt samples.")
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="answers_v2j-vectors-to-jokes-llama.json",
        help="Path to the test output file.")
    args = parser.parse_args()

    with open(args.prompts_path) as f:
        prompts = json.load(f)

    model, tokenizer = load_model(base_model, lora_weights, load_8bit)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        no_repeat_ngram_size=3
    )

    torch.manual_seed(SEED)
    answers = []
    i = 0
    for p in prompts:
        is_multiple_choice = "choices" in p and p["choices"]
        if is_multiple_choice:
            prompt = create_prompt(p["question"], p["choices"])
        else:
            prompt = create_prompt(p["question"])
        
        response = generate_response(model, tokenizer, prompt, generation_config, max_new_tokens)
        answers.append({
            "guid" : p["guid"],
            "model_answer": response
        })

        i+=1
        print(f"Prompt {i}")
        print(response)
        print("".join(["-"*30]))

    with open(args.output_path, "w") as f:
        json.dump(answers, f, indent=4)

