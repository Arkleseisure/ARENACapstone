"""
tiny_transformer_standard.py
Quick demo of a small Transformer encoder stack using PyTorch's built-ins.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# --- synthetic dataset ---
class SyntheticSequenceDataset(Dataset):
    def __init__(self, length: int, seq_len: int, vocab_size: int):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq = torch.ones((self.seq_len,), dtype=torch.long)#torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)
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

# --- quick training loop ---
if __name__ == "__main__":
    num_epochs = 100
    vocab_size = 5
    seq_len = 16
    d_model = 4
    nhead = 2
    num_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SyntheticSequenceDataset(2000, seq_len, vocab_size)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = TinyTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            T = src.size(1)
            mask = generate_square_subsequent_mask(T, device)
            logits = model(src, src_mask=mask)
            loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    total_time = time.time() - start_time
    print('Total time taken:', total_time)
    print(f'Time per epoch:', total_time/num_epochs)

