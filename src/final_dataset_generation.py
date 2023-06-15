import json
import os
import random

SEED = 42

DATA_DIR = "../data"

reddit_dataset_path = os.path.join(DATA_DIR, 'reddit_raw_dataset.json')
with open(reddit_dataset_path) as f:
    reddit_dataset = json.load(f)

reddit_dataset_by_question = {}
for datapoint in reddit_dataset:
    question = datapoint["question"]
    label = datapoint["label"]
    
    if not question in reddit_dataset_by_question:
        reddit_dataset_by_question[question] = {
            "good" : [],
            "bad" : []
        }

    reddit_dataset_by_question[question][label].append(datapoint)


random.seed(SEED)

# Add Reddit samples to dataset
reward_dataset = []
gen_dataset = []
i = 1
for question, datapoints in reddit_dataset_by_question.items():
    goods = datapoints["good"]
    bads = datapoints["bad"]
    
    to_sample = min(len(goods), len(bads))
    if to_sample == 0:
        continue
    
    for _ in range(to_sample):
        good = random.choice(goods)
        bad = random.choice(bads)
        reward_dataset.append({
            "entry_id" : f"r{i}",
            "query" : question,
            "chosen" : good["answer"],
            "rejected" : bad["answer"]
        })
        gen_dataset.append({
            "entry_id" : f"gr{i}",
            "instruction" : question,
            "output" : good["answer"]
        })
        i += 1

# Add ChatGPT interactions to reward dataset
chatgpt_dataset_path = os.path.join(DATA_DIR, "gpt_interactions.json")
with open(chatgpt_dataset_path, "r") as f:
    chatgpt_dataset = json.load(f)

for q in chatgpt_dataset:
    question = q["question"]
    
    is_open_ended = "choices" not in q or not q["choices"]
    if not is_open_ended:
        choices = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(q["choices"]))
        question += "\n" + choices
    
    reward_dataset.append({
        "entry_id" : f"e{i}",
        "query" : question,
        "chosen" : q["gpt_chosen"],
        "rejected" : q["gpt_rejected"]
    })
    
    gen_entry = {
        "entry_id" : f"ge{i}",
        "instruction" : q["question"]
    }

    if not is_open_ended:
        gen_entry["input"] = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(q["choices"]))
        
    gen_entry["output"] = q["gpt_chosen"]

    gen_dataset.append(gen_entry)
    i+=1

with open('../gen_dataset_v2j-vectors-to-jokes.json', 'w') as f:
        json.dump(gen_dataset, f, indent=4)
        
with open('../reward_dataset_v2j-vectors-to-jokes.json', 'w') as f:
        json.dump(reward_dataset, f, indent=4)