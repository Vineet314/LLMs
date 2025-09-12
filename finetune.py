import json
import math
import torch
import tiktoken

from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from model import LLM, LLMconfig, BlockConfig
from torch.nn.utils.rnn import pad_sequence 
from dataclasses import dataclass
from typing import Literal

data_path = "data/open_assistant/data.jsonl"

# needed to load checkpoint
@dataclass
class Trainconfig:
    dataset : str | Literal['shakespeare', 'tinystories', 'fineweb', 'wikitext']
    total_batch_size : int
    batch_size : int
    max_iters : int
    eval : bool
    eval_interval : int
    eval_iters : int
    learning_rate : float
    warmup_steps : int
    grad_clip : int
    compile : bool #= False if os.name != 'posix' else True
    save_model : bool
    ckpt_interval : int
    file_name : str
    act_recomp : bool
    wandb_log : bool
    wandb_project : str
    wandb_run_name : str

@dataclass
class FineTuneConfig:
    base_model : str
    data_path : str
    batch_size : int
    max_iters : int
    learning_rate : float
    lora_r : int
    lora_alpha : int

def load_pretrained_model(checkpoint_path, device):
    # print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_config']
    model = LLM(model_config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    
    return model, model_config

# --- Tokenizer and Special Tokens ---

def get_tokenizer(new_special_tokens):
    # Use the gpt2 tokenizer as requested
    enc = tiktoken.get_encoding("gpt2")
    
    # The gpt2 tokenizer has 50257 tokens. New tokens will start after that.
    # The <|endoftext|> token is at index 50256.
    special_tokens_dict = {name: 50257 + i for i, name in enumerate(new_special_tokens)}
    
    tokenizer = tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **special_tokens_dict}
    )
    return tokenizer

SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|endofturn|>"]
tokenizer = get_tokenizer(SPECIAL_TOKENS)

USER_TOKEN_ID = tokenizer.encode_single_token("<|user|>")
ASSISTANT_TOKEN_ID = tokenizer.encode_single_token("<|assistant|>")
EOT_TOKEN_ID = tokenizer.encode_single_token("<|endofturn|>")
PAD_TOKEN_ID = EOT_TOKEN_ID

def prepare_sample(conversation):
    input_ids, labels = [], []
    for msg in conversation["messages"]:
        role_token = USER_TOKEN_ID if msg["role"] == "user" else ASSISTANT_TOKEN_ID
        content_tokens = tokenizer.encode(msg["text"])
        
        input_ids.extend([role_token] + content_tokens + [EOT_TOKEN_ID])
        
        if msg["role"] == "user":
            # Mask user's prompt, role token, and end-of-turn token
            labels.extend([-1] * (len(content_tokens) + 2))
        else:
            # Mask role token and end-of-turn token, but NOT the content
            labels.extend([-1] + content_tokens + [-1])
            
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class ChatDataset(Dataset):
    def __init__(self, jsonl_path, prepare_fn):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.prepare_fn = prepare_fn

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.prepare_fn(self.data[idx])

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN_ID)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return inputs_padded, labels_padded


if __name__ == '__main__':
    config = FineTuneConfig(
        base_model= "demo_model_best.pt",
        data_path="data/open_assistant/data.jsnol",
        batch_size=2,
        max_iters=2000,
        learning_rate=5e-4,
        lora_r=16,
        lora_alpha=32)
    
    model, model_config = load_pretrained_model(config.base_model, "cuda")
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
        target_modules=["c_attn", "c_proj", "W_uq", "W_o", "lm_head"], # Adjust based on your model's layer names
        lora_dropout=0.05, bias="none")
    
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    model.to("cuda")
    
    # 5. Setup Dataloader
    print("Setting up dataloader...")
    train_dataset = ChatDataset(config.data_path, prepare_sample)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 6. Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1, fused=True)
    model.train()
    
    print("Starting fine-tuning...")
    iter_num = 0
    for epoch in range(5): # A simple epoch loop
        for step, (x, y) in enumerate(train_loader):
            # The dataloader handles padding, but we might need to truncate if a single example is too long
            # This now uses the NEW, larger block size
            x = x[:, :model_config.block_size].to("cuda")
            y = y[:, :model_config.block_size].to("cuda")

            # Forward pass
            logits, loss, _ = model(x, targets=y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if iter_num % 10 == 0:
                print(f"Iter {iter_num}: Loss = {loss.item():.4f}")
            
            iter_num += 1
            if iter_num >= config.max_iters:
                break
        if iter_num >= config.max_iters:
            break
            
    # 7. Save LoRA Adapters
    print("Saving LoRA adapters...")
    model.save_pretrained("lora_adapters_extended")
    print("Fine-tuning complete!")