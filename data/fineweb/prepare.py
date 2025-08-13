'''This script prepares the requested dataset. Needs to be run only once for a dataset.'''

import os
import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

n_proc = 4 + os.cpu_count() // 2   # 4 + 16//2 = 12
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
remote_name = "sample-10BT"
EOT_TOKEN_ID = 50256 # End-of-text token for gpt2

def tokenize_and_save():
    """
    Downloads, tokenizes, and saves the requested dataset.
    This version uses the `map` method for efficient tokenization.
    """

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, name=remote_name)

    split_dataset = dataset['train'].train_test_split(test_size=0.01, seed=1729, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename 'test' to 'val'

    enc = tiktoken.get_encoding("gpt2")

    def tokenize(example):
        """Tokenization function to be applied with `map`."""
        ids = enc.encode_ordinary(example['text'])
        ids.append(EOT_TOKEN_ID)  # Add End-of-Text token
        return {'ids': ids, 'len': len(ids)}

    # Tokenize parallel
    tokenized = split_dataset.map(
        tokenize,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=n_proc)

    # Save the tokenized datasets to binary files using np.memmap
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)  # total tokens
        file_path = f'{split}.bin'
        dtype = np.uint16  # GPT-2 tokenizer has 50257 tokens

        print(f"Saving {split} split to {file_path} ({arr_len:,} tokens)")

        # Create a memmap array on disk
        mmapped_arr = np.memmap(file_path, dtype=dtype, mode='w+', shape=(arr_len,))

        # Write into the memmap in chunks without holding all in RAM
        idx = 0
        for ids in tqdm(dset['ids'], desc=f"Writing {split}", total=len(dset)):
            ids_arr = np.array(ids, dtype=dtype)
            mmapped_arr[idx:idx + len(ids_arr)] = ids_arr
            idx += len(ids_arr)

        # Ensure data is flushed to disk
        mmapped_arr.flush()
        del mmapped_arr  # Close the memmap file

        print(f"Saved {arr_len:,} tokens to {file_path}")

    '''
    # if you have a crap ton of RAM: 
    # Save the tokenized datasets to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        file_path = f'{split}.bin'
        dtype = np.uint16  # GPT-2 tokenizer has 50257 tokens

        print(f"Saving {split} split to {file_path}")

        # Convert all token lists into one flat NumPy array
        all_tokens = np.concatenate([np.array(ids, dtype=dtype) for ids in dset['ids']])

        # Single write
        with open(file_path, 'wb') as f:
            f.write(all_tokens.tobytes())

        print(f"Saved {arr_len:,} tokens to {file_path}")
    '''


if __name__ == "__main__":
    tokenize_and_save()
    print("Data preparation complete.")