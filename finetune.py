import json
import math
import torch
import tiktoken

from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from model import LLM, BlockConfig
from torch.nn.utils.rnn import pad_sequence 
from dataclasses import dataclass, asdict
from typing import Literal, Optional

data_path = "data/open_assistant/data.jsonl"

# needed to load checkpoint
@dataclass
class BlockConfig:
    # global
    n_embd: int
    pos_emb: str | Literal['learn', 'sin', 'rope']
    dropout: float
    # attn
    attn: str | Literal['mha', 'mqa', 'gqa', 'mla']
    n_head: int
    # ffn
    moe: bool
    up_dim: int
    non_linearity: str | Literal['elu', 'lrelu', 'relu', 'gelu', 'swish', 'mish', 'silu','selu', 'celu', 'tanh', 'swiglu', 'sigmoid']

    # Optional fields with default as None
    # attn
    n_kv_heads: Optional[int] = None
    q_latent_dim:  Optional[int] = None
    kv_latent_dim: Optional[int] = None
    rope_head_dim: Optional[int] = None
    # ffn
    n_exp:    Optional[int] = None
    n_shared: Optional[int] = None
    n_act:    Optional[int] = None
    coeff:    Optional[float] = None
    aux_free: Optional[bool] = None
    alpha:    Optional[float] = None
    gamma:    Optional[float] = None

@dataclass
class LLMconfig:
    # token params
    vocab_size : int
    block_size : int
    n_embd : int
    pos_emb : str | Literal['learn','sin','rope']

    # model params
    dropout : float
    n_layer : int
    norm : str | Literal['layer','rms']

    act_recomp : bool
    CUSTOM_LAYERS : bool
    layer_configs : list[BlockConfig] | None

    model_type:str = "custom_model"
    _name_or_path: Optional[str] = None # ADD THIS LINE

    def get(self, attr):
        return getattr(self, attr, None)

    def __contains__(self, key): # ADD THIS METHOD
        """Enables the 'in' operator for LLMconfig instances."""
        if key == "_name_or_path":
            return True # PEFT checks for this specific key
        return hasattr(self, key)
    
    def __getitem__(self, key): # ADD THIS METHOD
        """Enables dictionary-style access, e.g., config["_name_or_path"]."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in LLMconfig for dictionary-style access.")

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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    ModelConfig:LLMconfig = checkpoint['model_config']
    new_config = LLMconfig(
        vocab_size=ModelConfig.vocab_size,
        block_size=ModelConfig.block_size,
        n_embd=ModelConfig.n_embd,
        pos_emb=ModelConfig.pos_emb,
        dropout=ModelConfig.dropout,
        n_layer=ModelConfig.n_layer,
        norm=ModelConfig.norm,
        act_recomp=ModelConfig.act_recomp,
        CUSTOM_LAYERS=ModelConfig.CUSTOM_LAYERS,
        layer_configs=ModelConfig.layer_configs,
        model_type="custom",
        _name_or_path="custom" # ADD THIS LINE
    )
    model = LLM(new_config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model, new_config

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
        content_tokens = tokenizer.encode(str(msg["text"]))
        
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

class PeftLLMWrapper(torch.nn.Module):
    def __init__(self, original_llm_instance:LLM):
        super().__init__()
        self.base_model = original_llm_instance
        self.config = self.base_model.config # PEFT might access .config

        # Optionally, adapt other methods if you plan to use PEFT's generation utilities
        # For now, we only focus on the forward pass error.
        # If you want to use model.generate() from PEFT, you'll need to wrap/adapt that too.

    # This forward method will be called by PeftModel
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        # Translate PEFT's input_ids to your LLM's idx
        # Translate PEFT's labels to your LLM's targets
        # Ignore attention_mask as your original LLM doesn't use it in forward
        
        # Pass other kwargs (like kv_caches) directly
        kv_caches = kwargs.get("kv_caches")

        # Call the original LLM's forward method
        return self.base_model.forward(idx=input_ids, targets=labels, kv_caches=kv_caches)

    # If you intend to use PeftModel's generation utilities, you'd also need
    # to adapt prepare_inputs_for_generation and potentially generate.
    # For instance, the original LLM.prepare_inputs_for_generation expects 'idx' not 'input_ids'.
    # For now, let's assume you're only focused on the training loop's forward call.
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        # PEFT will call this with 'input_ids', but your LLM expects 'idx'
        # Also, filter out 'attention_mask' if LLM.generate doesn't use it.
        kwargs.pop("attention_mask", None) # Ensure attention_mask is not passed to original LLM
        return self.base_model.prepare_inputs_for_generation(idx=input_ids, **kwargs)

    def generate(self, input_ids: torch.Tensor, **kwargs):
        # PEFT might call this with 'input_ids'. Your LLM expects 'idx'.
        return self.base_model.generate(idx=input_ids, **kwargs)

if __name__ == '__main__':
    config = FineTuneConfig(
        base_model= "demo_model_best.pt",
        data_path="data/open_assistant/data.jsonl",
        batch_size=2,
        max_iters=2500,
        learning_rate=9e-4,
        lora_r=16,
        lora_alpha=32)
    
    model, ModelConfig = load_pretrained_model(config.base_model, "cuda")
    model = PeftLLMWrapper(model)
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
        target_modules=["W_dq", "W_uq", "W_dkv", "W_uk", "W_uv", "lm_head", "c_proj", "c_fc"], # Adjust based on your model's layer names
        lora_dropout=0.05, bias="none")
    
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    model.to("cuda")
    
    # 5. Setup Dataloader
    train_dataset = ChatDataset(config.data_path, prepare_sample)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 6. Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1, fused=True)
    model.train()
    
    iter_num = 0
    for epoch in range(5): # A simple epoch loop
        for step, (x, y) in enumerate(train_loader):
            # The dataloader handles padding, but we might need to truncate if a single example is too long
            # This now uses the NEW, larger block size
            x = x[:, :ModelConfig.block_size].to("cuda")
            y = y[:, :ModelConfig.block_size].to("cuda")

            # Forward pass
            logits, loss, _ = model(input_ids=x, labels=y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            print(f"Iter {iter_num}: Loss = {loss.item():.4f}")
            
            iter_num += 1
            if iter_num >= config.max_iters:
                break
            if iter_num%100==0: model.save_pretrained("lora_adapters_extended")
        if iter_num >= config.max_iters:
            break
            
    # 7. Save LoRA Adapters
    print("Saving LoRA adapters...")
    model.save_pretrained("lora_adapters_extended")
    print("Fine-tuning complete!")