'''
This script loads a pre-trained LLM model and generates text from it.
It offers better argument parsing and a more robust way to load the model.

IMPORTATNT : For some reason, the model you want to run, should be on the same dir as this file. Idk why, will fix this later.

To run this script:
python model_sample.py --model_path=2_flash/flash_llm_model.pt --max_new_tokens=500 --start_text="Hello, world" --device="cuda"
or run ./sample.sh
'''

import os
import sys
import torch
import argparse
import tiktoken
from time import time

class LLMconfig : pass
class config : pass

# Add the parent directory to the sys.path to find the 'models' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    """Parses command-line arguments for the sampling script."""
    parser = argparse.ArgumentParser(description='Load a pre-trained LLM model and sample text.')
    parser.add_argument('--model_path',     type=str,   default=None,help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--max_new_tokens', type=int,   default=500,    help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature',    type=float, default=1.0,    help='Sampling temperature. Higher values make output more random.')
    parser.add_argument('--top_k',          type=int,   default=None,   help='Top-k sampling: sample from the top k most probable tokens.')
    parser.add_argument('--start_text',     type=str,   default="\n",   help='Initial text to prime the generation.')
    parser.add_argument('--device',         type=str,   default='cuda' if torch.cuda.is_available() else 'cpu',help='Device to use for generation (cpu or cuda).')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model with torch.compile()')
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    if args.model_path is None:
        print("No model path provided, using 6_sinusoidal_gqa/mqa_gqa_llm_model.pt")
        model_path = "6_sinusoidal_gqa/mqa_gqa_llm_model.pt"
    # Determine the device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = torch.load(model_path, map_location=device, weights_only=False)

    if args.compile:
        print("\ncompiling model...")
        model = torch.compile(model)
    model.eval() # Set model to evaluation mode

    enc = tiktoken.get_encoding('gpt2')

    start_ids = enc.encode(args.start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    t = time()
    with torch.no_grad():
        generated_ids = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    torch.cuda.synchronize()
    dt = time() - t

    generated_text = enc.decode(generated_ids[0].tolist())

    print(f'\ngeneration complete')
    print(f'\n\n--------------------------------------\n\nTime taken to generate = {dt:.2f}s')

if __name__ == '__main__':
    main()