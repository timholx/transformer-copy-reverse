import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(D)

        # only attend to the current position and earlier ones
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        out = attn_weights @ v
        out = self.w_o(out)
        return out, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=64, d_ff=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.block = TransformerBlock(d_model, d_ff)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.token_emb(idx) + self.pos_emb(positions)
        x, attn_weights = self.block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, attn_weights
