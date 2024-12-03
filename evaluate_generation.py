import argparse
import json
import torch
import statistics
import os
# os.chdir('../')

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, 
    AutoConfig,
)

from model import V2JRewardModel, V2JRewardModelConfig

def load_json(filename):
    """Load json file"""
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data


def save_dictlist_to_json(mydictlist, filename):
    """Save a list of dictionaries to json file"""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4) 
    f.close()

def create_prompt(question, choices=None):
    if choices:
        choices_str = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(choices))
        return f"Below is a question along with the available answer choices. Write a response that succinctly answers the question by selecting the correct choice or choices. Provide a reasoning.\n\n### Question:\n{question}\n\n### Choices:\n{choices_str}\n\n### Response:\n"
    return f"Below is a question. Write a response that succinctly answers the question and provides reasoning.\n\n### Question:\n{question}\n\n### Response:\n"

class TestDataset(Dataset):
    """Simple dataset module for testing the reward model"""
    def __init__(self, test_ds, pred_ds):
        self.ds = test_ds
        self.pred_ds = pred_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ix):
        d = self.ds[ix]
        prompt = create_prompt(d['question'], choices=d['choices'] if 'choices' in d.keys() else None)
        p = self.pred_ds[ix]
        assert d['guid'] == p['guid']
        return {
            "chosen" : "Human: {}\nAssistant: {}".format(prompt, p["model_answer"]),
            "rejected": ""
        }


class Reward(torch.nn.Module):
    """
    Wrapper class for the reward model, 
    which handles loading the model and tokenizers, 
    and the forward pass for final predictions
    """
    def __init__(self, model_path, device):
        super().__init__()

        # Load student-defined reward model and its associated config
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)
        
        # Move model to the given device
        self.device = device
        self.model = self.model.to(self.device)

    def check_reward_type(self, rewards):
        return isinstance(rewards, list) and all(isinstance(r, dict) for r in rewards)

    def forward(self, demonstrations):
        """
        Get the reward predictions from student's reward model
        Args:
            demonstrations: list of dicts in the format of 
            {'chosen': str, 'rejected': str}
        Return:
            rewards: list of dicts in the format of
            {'chosen': float, 'rejected': float} 
        """
        # ===== Get the rewards from student's reward model =====
        rewards = self.model.get_rewards(demonstrations, self.device)

        # ===== Check the reward format =====
        assert self.check_reward_type(rewards), "The rewards must be a list of dicts"
        assert len(rewards) == len(demonstrations), "The number of rewards must match the number of demonstration pairs"
        return rewards
    

class Evaluator:
    def __init__(self, model_path, ds_test, ds_pred, device):
        # Load the model and dataset
        self.load_model(model_path, device)
        self.ds_test = ds_test
        self.ds_pred = ds_pred
        self.dataset = TestDataset(ds_test, ds_pred)
        self.dataloader = DataLoader(
            self.dataset, batch_size=2, shuffle=False,
            collate_fn=lambda x: x)

    def load_model(self, model_path, device):
        """Load the reward model from the specified path"""
        self.model = Reward(model_path, device)
    
    def evaluate(self):
        """Evaluate the model on the test dataset"""
        rewards = []
        for batch in tqdm(self.dataloader):
            rewards.extend(self.model(batch))

        # ===== Check the rewards by doing pair-wise ranking =====
        #num_correct = sum(reward['chosen'] > reward['rejected'] for reward in rewards)
        #acc = num_correct / len(self.ds_test)
        seq_rewards = [reward['chosen'] for reward in rewards]
        print(f"Evaluation Complete, Mean: {statistics.mean(seq_rewards)}, Std: {statistics.stdev(seq_rewards)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/v2j-reward-large",
        help="Path to the reward model")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="prompts.json",
        help="Path to the test dataset")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="answers_v2j-vectors-to-jokes.json",
        help="Path to the model predictions"
    )
    args = parser.parse_args()
    
    AutoConfig.register('V2JRewardModel', V2JRewardModelConfig)
    AutoModel.register(V2JRewardModelConfig, V2JRewardModel)
    
    reward_dataset = load_json(args.data_path)
    pred_dataset = load_json(args.predictions_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator(args.model_path, reward_dataset, pred_dataset, device=device)
    evaluator.evaluate()