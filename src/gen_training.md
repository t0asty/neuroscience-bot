# Generative model training

Download [alpaca-lora](https://github.com/tloen/alpaca-lora) code and checkout to the given commit.
```
git clone https://github.com/tloen/alpaca-lora.git
cd alpaca-lora
git checkout 8bb8579
```

Install dependencies from `requirements.txt`.
```
pip install -r requirements.txt
```

In order for bitsandbytes (quantization) to work, add your Python lib path to `LD_LIBRARY_PATH`.
```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/
```

Copy the provided src/v2j_template.json to alpaca-lora/templates/v2j_template.json.
```
cp ../src/v2j_template.json templates/v2j_template.json
```

Run LoRA fine-tuning with using the following command. For simplicity, we are using LLaMA 7B weights uploaded to Huggingface (`eachadea/vicuna-7b-1.1`). The training is performed on the generative dataset we created. We follow hyperparameters used by LoRA and Alpaca papers. For the prompt, we use created v2j template.
```
python finetune.py \
    --base_model 'eachadea/vicuna-7b-1.1' \
    --data_path '../gen_dataset_v2j-vectors-to-jokes.json' \
    --output_dir '../models/v2j-vicgen-lora' \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 100 \
    --prompt_template_name 'v2j_template' \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'
```