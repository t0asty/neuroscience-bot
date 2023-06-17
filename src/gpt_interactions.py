import gpt_wrapper
from gpt_wrapper.chat import Chat
import json
import os
import random
from tqdm import tqdm

DATA_DIR = "../data"

with open("api_key.txt", "r") as f:
    api_key = f.readline()
gpt_wrapper.api_key = api_key

path = os.path.join(DATA_DIR, "solutions_v1.json")
with open(path, "r") as f:
    questions = json.load(f)

# Take only questions with answer
questions_with_answers = []
for q in questions:
    if not "answer" in q or not q["answer"]:
        continue
    questions_with_answers.append(q)

train_size = 600
gen_test_size = 200

random.seed(124)
random.shuffle(questions_with_answers)

train_questions = questions_with_answers[:train_size]
gen_test_questions = questions_with_answers[train_size:train_size+gen_test_size]

instruction_open_ended = "You will be given a question and a correct answer. Do not mention having correct answer."
instruction_multiple_choice = "You will be given a question, available answers and the correct ones. Do not mention having correct answer."
instruction_good = "Act as a science teacher that answers the question correctly and provides a short reasoning and explanation to improve student's learning experience. Answer in the following format: The answer is/answers are ... because ..."
instruction_bad = "Act as a bad science teacher that answers the question by providing bad reasoning, incorrect answers or the reasoning that students cannot understand. As a bad teacher, you are not improving student's learning experience, but you should still sound professional. Answer in the following format: The answer is/answers are ... because ..."

path = os.path.join(DATA_DIR, "gen_test_questions.json")
with open(path, "w") as f:
    json.dump(gen_test_questions, f, indent=4)

k=0
generated_samples = []
for q in train_questions:
    q = q.copy()
    is_open_ended = "choices" not in q or not q["choices"]
    
    base_instruction = instruction_open_ended if is_open_ended else instruction_multiple_choice
    good_instruction = base_instruction + " " + instruction_good
    bad_instruction = base_instruction + " " + instruction_bad
    
    prompt = q["question"] + "\n"
    if not is_open_ended:
        choices = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(q["choices"]))
        prompt += choices

    correct = str(q["answer"])
    if correct.startswith("[") and correct.endswith("]"):
        correct = correct[1:-1]
    prompt += "\n\nCorrect: " + correct

    if "explanation" in q and q["explanation"]:
        prompt += "\nExplanation: " + str(q["explanation"])
    
    chat_good = Chat.create("synthetic dataset good")
    chat_bad = Chat.create("synthetic dataset bad")
    
    message_good = chat_good.ask(prompt, instruction=good_instruction)
    message_bad = chat_bad.ask(prompt, instruction=bad_instruction)
    
    q["gpt_chosen"] = str(message_good)
    q["gpt_rejected"] = str(message_bad)
    
    generated_samples.append(q)
    
    path = os.path.join(DATA_DIR, "gpt_interactions.json")
    with open(path, "w") as f:
        json.dump(generated_samples, f, indent=4)
    
    k+=1
    print(k)