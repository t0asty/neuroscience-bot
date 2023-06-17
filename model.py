from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    DebertaV2Config,
    AutoModelForSequenceClassification
)
import torch

class V2JRewardModelConfig(DebertaV2Config):
    """
    Our CustomRewardModel is a fine-tuned OpenAssistant/reward-model-deberta-v3-large-v2.
    Therefore, it's config class inherits DebertaV2Config.
    """
    model_type="V2JRewardModel"
    
class V2JRewardModel(PreTrainedModel):
    config_class = V2JRewardModelConfig
    MAX_SEQ_LEN = 512

    def __init__(self, config):
        super().__init__(config)

        hf_pretrained_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        #hf_pretrained_model_name = "OpenAssistant/reward-model-deberta-v3-base"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_pretrained_model_name)
        self.config = config

    def forward(self, inputs):
        return self.model(**inputs)

    def get_rewards(self, demonstrations, device=None):
        """
        Get the rewards for the demonstrations
        Args:
            demonstrations: list of dicts in the format of
            {'chosen': str, 'rejected': str}
        Return:
            rewards: list of dicts in the format of
            {'chosen': float, 'rejected': float} 
        """
        back_to_train = self.model.training
        self.model.eval()
        with torch.no_grad():
            rewards = []
            for pair in demonstrations:
                encoded_chosen = self.tokenizer(
                    pair['chosen'], return_tensors="pt",
                    truncation=True, max_length=self.MAX_SEQ_LEN)
                encoded_reject = self.tokenizer(
                    pair['rejected'], return_tensors="pt",
                    truncation=True, max_length=self.MAX_SEQ_LEN)
                
                if device:
                    encoded_chosen = encoded_chosen.to(device)
                    encoded_reject = encoded_reject.to(device)

                scores_chosen = self.forward(encoded_chosen)
                scores_reject = self.forward(encoded_reject)
                rewards.append({
                    'chosen': scores_chosen.logits.item(),
                    'rejected': scores_reject.logits.item()
                })
        if back_to_train:
            self.model.train()
        return rewards