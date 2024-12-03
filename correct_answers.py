import json

with open('prompts.json') as f:
    prompts = json.load(f) 

with open('chatgpt.json') as f:
    solutions = json.load(f)

correct = 0
for prompt in prompts:
    for solution in solutions:
        if prompt['guid'] == solution['guid']:
            print("===========================")
            print(prompt['question'])
            print("------------------------")
            print(solution['model_answer'])
            print("------------------------")
            print(prompt['answer'])
            corr = input("Correct? 1: True, 2: False")
            if corr == "1":
                correct += 1

print("Correct answers: ", correct)