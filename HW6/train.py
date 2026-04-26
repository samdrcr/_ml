import torch
from architecture import LanguageModel

# --- Hyperparameters ---
batch_size = 32
block_size = 64 # Context length
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
# -----------------------

# Load dataset (make sure you have a text file named 'input.txt')
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/Val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_subset = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_subset) - block_size, (batch_size,))
    x = torch.stack([data_subset[i:i+block_size] for i in ix])
    y = torch.stack([data_subset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = LanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the trained weights
torch.save(model.state_dict(), 'model_weights.pth')

# Test generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerated Text Output:")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))