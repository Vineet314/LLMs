'''This script builds and trains an LLM model based on the user's CLI inputs. Available settings to choose from : 
1. Attention Type (with or without KV caching): 
    - Multi Head Attention
    - Flash Attention based on Multi Head Attention (MHA)
    - Flash Attention based on Multi Query Attention (MQA)
    - Flash Attention based on Grouped Query Attention (GQA)
    - Multi Head Latent Attention (MHLA)
    - (Work in progress) Flash Multi Head Latent Attention

2. Positional Encodings:
    - Learnable PE
    - Sinusoidal PE
    - Rotary PE (RoPE)

The following arguments can be provided. If not, the default values are set : 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from dataclasses import dataclass 

@dataclass
class LLMconfig:
    vocab_size : int
    






    




    
