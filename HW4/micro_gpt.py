import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. Hyperparameters ---
batch_size = 32
block_size = 12  # Max name length
max_iters = 2000
lr = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
n_head = 4
n_layer = 2

# --- 2. Data Loading ---
with open('input.txt', 'r') as f:
    words = f.read().splitlines()

# Character mapping (Tokenizer)
chars = sorted(list(set(''.join(words))))
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0 # End of word / Padding token
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Build dataset of (context -> target)
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X, Y = torch.tensor(X), torch.tensor(Y)

# --- 3. The Transformer Components ---


class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention + Residual
        x = x + torch.cat([h(self.ln1(x)) for h in self.sa], dim=-1)
        # Feed-Forward + Residual
        x = x + self.ffwd(self.ln2(x))
        return x

class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# --- 4. Training ---
model = MicroGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("Training Micro GPT...")
for i in range(max_iters):
    ix = torch.randint(0, X.shape[0], (batch_size,))
    xb, yb = X[ix].to(device), Y[ix].to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i % 500 == 0: print(f"Step {i}: Loss {loss.item():.4f}")

# --- 5. Generate Names ---
print("\nGenerated Names:")
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        logits, _ = model(torch.tensor([context]).to(device))
        probs = F.softmax(logits[:, -1, :], dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix == 0: break
        out.append(itos[ix])
    print(''.join(out))