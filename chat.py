# chat.py

import torch
import argparse
import tiktoken
from model import LLM
from finetune import LLMconfig,Trainconfig, BlockConfig, FineTuneConfig
from peft import PeftModel

# NEW IMPORT
from transformers import GenerationConfig 

# --- NEW/MOVED: Tokenizer Setup and Special Token IDs ---
# Define SPECIAL_TOKENS as a LIST for the tokenizer's internal dictionary mapping
SPECIAL_TOKENS_LIST = ["<|user|>", "<|assistant|>", "<|endofturn|>"] 

def get_tokenizer(new_special_tokens):
    enc = tiktoken.get_encoding("gpt2")
    special_tokens_dict = {name: 50257 + i for i, name in enumerate(new_special_tokens)}
    tokenizer = tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={**enc._special_tokens, **special_tokens_dict}
    )
    return tokenizer

# Initialize the tokenizer globally in chat.py
tokenizer = get_tokenizer(SPECIAL_TOKENS_LIST)

# Define special token IDs globally for generation config and chat function
USER_TOKEN_ID = tokenizer.encode_single_token("<|user|>")
ASSISTANT_TOKEN_ID = tokenizer.encode_single_token("<|assistant|>")
EOT_TOKEN_ID = tokenizer.encode_single_token("<|endofturn|>")
PAD_TOKEN_ID = EOT_TOKEN_ID # Using EOT as pad token as in finetune.py
# --- END NEW/MOVED BLOCK ---

# --- Main Chat Function ---
def chat(args, model, tokenizer_instance, device): # Renamed tokenizer to tokenizer_instance to avoid conflict
    """
    Handles the interactive chat loop.
    """
    # Define SPECIAL_TOKENS_SET here or ensure it's passed as an argument.
    # Defining it as a SET here to directly address the error.
    SPECIAL_TOKENS_SET = set(SPECIAL_TOKENS_LIST) # Use the global list

    while True:
        try:
            # 1. Get user input
            user_prompt = input("\nUser: ")

            # 2. Check for exit condition
            if user_prompt.lower() in ["exit", "quit"]:
                print("\nExiting chatbot. Goodbye! 👋")
                break
            
            # Handle empty input
            if not user_prompt.strip():
                continue

            # 3. Prepare the prompt in the chat format
            prompt_string = f"<|user|>{user_prompt}<|endofturn|><|assistant|>"
            
            # --- FIX: Pass allowed_special as a SET to tokenizer.encode ---
            start_ids = tokenizer_instance.encode(prompt_string, allowed_special=SPECIAL_TOKENS_SET)
            
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

            # 4. Generate the response
            print("\nAssistant: ", end="", flush=True)
            with torch.no_grad():
                
                # Call the generate function
                y = model.generate(x,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature,
                                   topk=args.top_k,
                                   EOT=EOT_TOKEN_ID) # Pass EOT_TOKEN_ID to stop generation
                
                generated_tokens = y[0].tolist()
                
                # Decode and clean up the response
                full_text = tokenizer_instance.decode(generated_tokens)
                assistant_response = full_text.split('<|assistant|>')[-1].replace('<|endofturn|>', '').strip()

                print(assistant_response) 

        except KeyboardInterrupt:
            print("\nExiting chatbot. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

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
        _name_or_path="custom"
    )
    model = LLM(new_config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model, new_config

# --- Main Setup Function ---
def main(args):
    """
    Loads the model and tokenizer, then starts the chat loop.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    base_model, ModelConfig = load_pretrained_model(args.base_model_path, "cuda") 

    # --- NEW: Manually attach generation_config to the base_model ---
    # This ensures the LLM instance itself has generation_config, as expected by PeftModel's generate
    base_model.generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, 
        top_k=args.top_k,
        pad_token_id=PAD_TOKEN_ID, # Use the globally defined PAD_TOKEN_ID
        eos_token_id=EOT_TOKEN_ID, # Use the globally defined EOT_TOKEN_ID
        do_sample=True if args.temperature > 0.0 else False, # Enable sampling if temperature > 0
    )
    # --- END NEW BLOCK ---

    model = PeftModel.from_pretrained(base_model, args.lora_adapters_path)
    model.eval()  # Set the model to evaluation mode

    # --- Start the interactive chat ---
    # Pass the globally initialized tokenizer
    chat(args, model, tokenizer, device) 

# ... (rest of argparse and if __name__ == '__main__': block)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a CLI chatbot using a fine-tuned LoRA model.")
    parser.add_argument('--base_model_path',  type=str, default='demo_model_best.pt', help='Path to the original pre-trained model checkpoint.')
    parser.add_argument('--lora_adapters_path', type=str, default='lora_adapters_extended', help='Path to the saved LoRA adapters directory.')
    parser.add_argument('--max_new_tokens',   type=int, default=500, help='Maximum number of new tokens to generate per turn.')
    parser.add_argument('--temperature',      type=float, default=0.9, help='Generation temperature.')
    parser.add_argument('--top_k',            type=int, default=100, help='Top-k sampling.')
    
    args = parser.parse_args()
    main(args)