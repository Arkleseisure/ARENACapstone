"""
tiny_transformer_standard.py
Quick demo of a small Transformer encoder stack using PyTorch's built-ins.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from random import randint
import wandb
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- synthetic dataset ---
class SyntheticSequenceDataset(Dataset):
    def __init__(self, length: int, seq_len: int, vocab_size: int):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        a = randint(2, 4)
        if a == 2:
            pattern = [1, 1, 0]
        elif a == 3:
            pattern = [0, 0, 1]
        else:
            pattern = None

        arr = [5, a]
        for i in range(self.seq_len - 2):
            if pattern:
                arr.append(pattern[i % len(pattern)])
            else:
                arr.append(randint(0, 1))
        seq = torch.tensor(arr)
        #seq = torch.ones((self.seq_len,), dtype=torch.long)
        #torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)
        src = seq[:-1]
        tgt = seq[1:]
        return src, tgt

# --- tiny model using standard API ---
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=64, d_model=32, nhead=4, num_layers=2, dim_ff=128, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))  # learnable
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True  # (B, T, C)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_mask=None):
        B, T = x.shape
        x = self.embedding(x) + self.pos_emb[:, :T]
        x = self.encoder(x, mask=src_mask)
        return self.head(x)

def generate_square_subsequent_mask(sz: int, device):
    """Causal mask for autoregressive training."""
    mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
    return mask

