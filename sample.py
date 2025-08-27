import torch
import argparse
import tiktoken
from time import perf_counter
from model import LLM, LLMconfig

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        print("Model checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    config = checkpoint['config']

    tokenizer = tiktoken.get_encoding("gpt2")

    model = LLM(config)
    model.to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    start_ids = tokenizer.encode(args.prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    
    with torch.no_grad():
        a = perf_counter()
        y = model.generate(x,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature,
                           topk=args.top_k)
        b = perf_counter()
        generated_text = tokenizer.decode(y[0].tolist())

        print('\n\n',generated_text,f'\n\ntime taken to genertate {args.max_new_tokens} tokens = {b-a:.3f}s')
        
    print("-" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text using a trained LLM model with tiktoken.")
    parser.add_argument('--model_path',     type=str,   default='llm_model.pt', help='Path to the saved model checkpoint (.pt file)')
    parser.add_argument('--prompt',         type=str,   default='\n',           help='The starting prompt for text generation.')
    parser.add_argument('--max_new_tokens', type=int,   default=300,            help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature',    type=float, default=0.1,            help='Generation temperature. >1.0 for more creative, <1.0 for more deterministic. Do not set to 0.')
    parser.add_argument('--top_k',          type=int,   default=200,            help='Top-k sampling. Limits sampling to the k most likely tokens.')
    
    args = parser.parse_args()
    main(args)
