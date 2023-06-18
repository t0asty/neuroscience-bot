import gpt_wrapper
from gpt_wrapper.chat import Chat
import json
import os
from tqdm import tqdm
import os
os.chdir('../')
import random
random.seed(42)

with open('api_key.txt', 'r') as f:
    api_key = f.readline()

gpt_wrapper.api_key = api_key

print(Chat.budget())

with open('prompts.json', 'r') as f:
    problems = json.load(f)

samples = []
problem_num = 0
print(len(problems))
for problem in problems:
    chat1 = Chat.create("Baseline run")

    is_open_ended = False
    if ('choices' not in problem.keys()) or (problem['choices'] is None):
        is_open_ended = True

    print("################################################################################")

    problem_num += 1
    print(f"{problem_num}: ")

    if is_open_ended:
        try:
            prompt = problem['question']
        except KeyError:
            continue
        print(prompt)
        message1 = chat1.ask(prompt, instruction=" You will be given a question. Act as a science teacher that answers the question correctly and provides a short reasoning and explanation to improve student's learning experience. You sound like you answered the question for the student.")
        print("===============================")
        print(message1)

    else:
        try:
            prompt = f"""
            {problem['question']}
            """

            choices = "\n".join(f"{ind}: {choice}" for ind, choice in enumerate(problem['choices']))

            prompt += choices
        except KeyError:
            continue
        print(prompt)
        message1 = chat1.ask(prompt, instruction=" You will be given a question and answer choices. Act as a science teacher that answers the question correctly and provides a short reasoning and explanation to improve student's learning experience. You sound like you answered the question for the student.")
        print("===============================")
        print(message1)

    problem = {
        'guid': problem['guid'],
        'model_answer': message1.content
    }
    samples.append(problem)

    with open('data/answers_chatgpt.json', 'w') as f:
        json.dump(samples, f, indent=4)