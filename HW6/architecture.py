import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionUnit(nn.Module):
    """ A single head of self-attention """
    def __init__(self, embedding_size, head_size, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Calculate attention scores ("affinities")
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        # Weighted aggregation of values
        v = self.value(x)
        out = weights @ v 
        return out

class MultiStrategyAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size, embedding_size, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionUnit(embedding_size, head_size, context_length, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class ProcessingBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, embedding_size, num_heads, context_length, dropout):
        super().__init__()
        head_size = embedding_size // num_heads
        self.attention = MultiStrategyAttention(num_heads, head_size, embedding_size, context_length, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_length, num_layers, num_heads, dropout):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(context_length, embedding_size)
        self.blocks = nn.Sequential(*[ProcessingBlock(embedding_size, num_heads, context_length, dropout) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_final(x) 
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx