# --- quick training loop ---
def train_model(vocab_size = 6,
    seq_len = 8,
    epochs = 500,
    batch_size = 16384,
    d_model = 3,
    nhead = 3,
    num_layers = 2,
    dim_ff = 2,
    lr = 1e-2,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_wandb=False,
    verbose=False
):

    train_ds = SyntheticSequenceDataset(batch_size, seq_len, vocab_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TinyTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, max_len=seq_len).to(device)
    if verbose:
        print('Model param count:', sum(p.numel() for p in model.parameters()))
        print('Model count by layer:')
        for p in model.named_parameters():
            print(p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    current_loss = 0

    if use_wandb:
        wandb.init(project='ARENACapstone', config={'vocab_size': vocab_size, 'seq_len': seq_len, 'epochs': epochs, 
                                                    'batch size': batch_size, 'd_model': d_model, 'n heads': nhead, 
                                                    'num layers': num_layers, 'd_mlp': dim_ff, 'lr': lr})
    if verbose:
        pbar = tqdm(total=epochs) 
    for epoch in range(epochs):
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

        current_loss = total_loss/len(train_loader)

        if verbose:
            pbar.set_description(f'loss: {current_loss}')
            pbar.update(1)
        if use_wandb:
            wandb.log({'loss':current_loss})

    return model

def test_model(model):
    print(model)
    train_ds = SyntheticSequenceDataset(batch_size, seq_len, vocab_size)
    # sample prediction
    for j in range(6):
        sample_src, _ = train_ds[j]
        sample_src = sample_src.unsqueeze(0).to(device)
        #sample_src = torch.tensor([[j for i in range(seq_len)]], device=device)
        mask = generate_square_subsequent_mask(sample_src.size(1), device)
        with torch.no_grad():
            logits = model(sample_src, src_mask=mask)
            preds = logits.argmax(-1)
        print("input :", sample_src.squeeze(0).tolist())
        print("preds :", preds.squeeze(0).tolist())
        for j in range(len(logits[0])):
            print(f'logits {j}:', logits.squeeze(0).tolist()[j])
            print(f'probs {j}:', torch.softmax(logits, -1).squeeze(0).tolist()[j])

def rl_model(base_model, finetuning_lr = 1e-4, rl_steps = 1000, rl_batch_size = 1024, beta = 0.1, eps = 1e-4, verbose=False): 
    model = TinyTransformer(vocab_size, d_model, nhead, num_layers, dim_ff, seq_len)   # same arch as base_model
    model.load_state_dict(base_model.state_dict())  # copy weights
    model = model.to(device)


    # --- reward: fraction of 1s in the sequence ---
    def reward_fn(output_tokens: torch.LongTensor):
        # output_tokens: (B, T)
        # Compare each token with the previous token
        # Shift tokens right and compare
        # Reward tokens that differ from their predecessor
        alternating_mask = output_tokens[:, 1:] != output_tokens[:, :-1]
        return alternating_mask.sum(dim=1).float()
        # rewards = (output_tokens == 0).mean(dim=1, dtype=float)
        # return rewards

    # --- roll out sequences from the current model ---
    def generate_sequences(model, batch_size, seq_len, vocab_size, device):
        model.eval()
        x = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # start token 0
        for t in range(seq_len-1):
            with torch.no_grad():
                mask = generate_square_subsequent_mask(x.size(1), device)
                logits = model(x, src_mask=mask)[:, -1]  # predict next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample
            x = torch.cat([x, next_token], dim=1)
        return x  # (B, seq_len)

    # --- log-probs of each sampled sequence under the current model ---
    def log_probs_of_sequence(model, seq, device):
        model.eval()
        B, T = seq.shape
        mask = generate_square_subsequent_mask(T-1, device)
        model = model.to(device)
        logits = model(seq[:, :-1], src_mask=mask)  # predict next token
        log_probs = torch.log_softmax(logits, dim=-1)
        #chosen = seq[:, 1:].unsqueeze(-1)
        #log_probs_taken = log_probs.gather(-1, chosen).squeeze(-1)  # (B, T-1)
        return log_probs  # log_probs of the sequence

    # --- simple GRPO / REINFORCE step ---
    optimizer = torch.optim.Adam(model.parameters(), lr=finetuning_lr)  # small lr for fine-tuning

    if verbose:
        pbar = tqdm(total=rl_steps)
        
    for step in range(rl_steps):  # number of RL steps
        seqs = generate_sequences(model, batch_size=rl_batch_size, seq_len=seq_len, vocab_size=vocab_size, device=device)
        rewards = reward_fn(seqs)                     # (B,)
        logp = log_probs_of_sequence(model, seqs, device)  # (B, T-1, V)
        p = torch.exp(logp)
        with torch.no_grad():
            logp_base = log_probs_of_sequence(base_model, seqs, device) # (B, T-1, V)


        kl_per_token = p * (logp - logp_base)  # (B, T-1, V)
        kl_per_token = kl_per_token.sum(-1) # (B, T-1)

        kl = kl_per_token.mean(-1) # (B,)

        # baseline to reduce variance
        baseline = rewards.mean().detach()
        advantages = (rewards - baseline)/(rewards.std().detach() + eps)

        # finds the logprobs which were actually taken
        chosen = seqs[:, 1:].unsqueeze(-1)
        logp_taken = logp.gather(-1, chosen).squeeze(-1)  # (B, T-1)
        logp = logp_taken.mean(-1)

        # print('logp:', logp)
        # print('kl:', kl)
        loss = (advantages * (-logp) + beta * kl).mean()  # maximize expected reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_reward = rewards.mean().item()
        avg_kl = kl.mean().item()
        if verbose:
            pbar.set_description(f'loss={loss.item():.4f}  avg_reward={avg_reward:.3f} KL={avg_kl:.3f}')
            pbar.update(1)

    return model

import numpy as np

def optimal_distribution(alpha: float):
    """
    Solve for optimal (a,b,c) and q* given alpha.
    Returns dict with a,b,c,q*, reward, KL, loss.
    """
    # Optimal bias for each bit in the C branch
    q_star = 1 / (1 + np.exp(-1/alpha))

    # Per-bit KL contribution under C
    KL_bit = q_star * np.log(2*q_star) + (1-q_star) * np.log(2*(1-q_star))

    # Unnormalized weights for A, B, C branches
    wA = np.exp(4/alpha)
    wB = np.exp(2/alpha)
    wC = np.exp((6*q_star + 6*KL_bit)/alpha)

    # Normalize
    Z = wA + wB + wC
    a, b, c = wA/Z, wB/Z, wC/Z

    # Compute reward
    reward = 4*a + 2*b + 6*c*q_star

    # Compute KL
    KL = (
        a*np.log(3*a) +
        b*np.log(3*b) +
        c*np.log(3*c) +
        6*c*KL_bit
    )

    # Loss
    loss = alpha*KL - reward

    return {
        "a": a,
        "b": b,
        "c": c,
        "q*": q_star,
        "reward": reward,
        "KL": KL,
        "loss": loss
    }

# Example usage
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
    sol = optimal_distribution(alpha)
    print(f"\nalpha={alpha}")
    for k,v in sol.items():
        print(f"{k}: {v:.4f}")


batch_size = 1024
rl_batch_size = 1024

test = torch.tensor([[5]], device=device)
model_probs = torch.tensor([[0 for i in range(6)]], dtype=float, device=device)
base_model_probs = torch.tensor([[0 for i in range(6)]], dtype=float, device=device)

def quick_test_model(model, model_probs):
    model_logits = model(test)
    p_m = torch.softmax(model_logits, dim=-1)
    return torch.cat((model_probs, p_m[0, :, :]))
    
'''
model = rl_model(base_model)
test_model(model)
test_model(base_model)
'''

vocab_size = 6
seq_len = 8
epochs = 500
batch_size = 1024
d_model = 3
nhead = 3
num_layers = 2
dim_ff = 2
lr = 1e-2
model = train_model(vocab_size, seq_len, epochs, batch_size, d_model, nhead, num_layers, dim_ff, lr, device)
base_model = TinyTransformer(vocab_size, d_model, nhead, num_layers, dim_ff, seq_len)   # same arch as base_model
base_model.load_state_dict(model.state_dict())  # copy weights
for p in base_model.parameters():
    p.requires_grad_(False)

base_model = base_model.to(device)
test_model(model)
test_model(base_model)

for i in tqdm(range(1000)):
    base_model = train_model(batch_size=batch_size)
    model = rl_model(base_model, rl_batch_size=rl_batch_size)

    model_probs = quick_test_model(model, model_probs)
    base_model_probs = quick_test_model(base_model, base_model_probs)

    model_means = model_probs[1:].mean(dim=0)
    base_model_means = base_model_probs[1:].mean(dim=0)
    model_stds = model_probs[1:].std(dim=0)
    base_model_stds = base_model_probs[1:].std(dim=0)

    text = f'Iteration: {i+1}\nModel mean:{model_means}\nModel std:{model_stds}\nBase Model means:{base_model_means}\nBase Model stds:{base_model_stds}'
    with open('log_file.txt', 'w') as f:
        f.write(text)




    
