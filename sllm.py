import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
from typing import Optional, Tuple, Dict, List
import os
import warnings
import pickle
warnings.filterwarnings('ignore')

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("Warning: 'tokenizers' library not available. Install with: pip install tokenizers")
    raise ImportError("HuggingFace tokenizers library is required. Install with: pip install tokenizers")

class HuggingFaceBPETokenizer:
    
    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        
    def train(self, corpus: str):
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library required. Install with: pip install tokenizers")
        
        print("Training HuggingFace BPE tokenizer...")
        
        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Set up trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2
        )
        
        # Write corpus to temporary file for training
        temp_file = "temp_corpus.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        
        # Train tokenizer
        self.tokenizer.train([temp_file], trainer)
        
        os.remove(temp_file)
        
        # Set up post-processing
        self.tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[("<bos>", 2), ("<eos>", 3)]
        )
        
        print(f"HuggingFace BPE training completed. Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def encode(self, text: str, add_special_tokens: bool = True):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode(self, token_ids: List[int]):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.decode(token_ids)
    
    def save(self, filepath: str):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        self.tokenizer.save(filepath)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):

        self.tokenizer = Tokenizer.from_file(filepath)
        print(f"Tokenizer loaded from {filepath}")
    
    @property
    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        
        # Create inverse frequencies for half the dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute frequencies
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create cos and sin tables for the full dimension
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        
        # Repeat each frequency for pairs 
        cos_table = torch.stack([cos_freqs, cos_freqs], dim=-1).flatten(-2)
        sin_table = torch.stack([sin_freqs, sin_freqs], dim=-1).flatten(-2)
        
        self.register_buffer('cos_table', cos_table)
        self.register_buffer('sin_table', sin_table)

    def rotate_half(self, x):
   
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: [batch, heads, seq_len, head_dim]
        cos = self.cos_table[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = self.sin_table[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        
        return x * cos + self.rotate_half(x) * sin

class SwiGLU(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q, T)
        k = self.rope(k, T)
        
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.feed_forward = SwiGLU(dim, 4 * dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm architecture
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class SmallLLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, n_layers: int = 6, 
                 n_heads: int = 4, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights (embeddings and output layer share weights)
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model has {total_params:,} parameters ({total_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = x.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask = mask.view(1, 1, T, T)
        
        # Token embeddings
        x = self.embedding(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss


class TextDataset(Dataset):
    def __init__(self, text_file: str, tokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Text length: {len(text)} characters")
        
        # Tokenize text
        self.data = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Get actual vocab size
        vocab_size = tokenizer.vocab_size_actual
            
        print(f"Dataset loaded: {len(self.data)} tokens, vocabulary size: {vocab_size}")
        print(f"Compression ratio: {len(text) / len(self.data):.2f}x")

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class TextGenerator:
    def __init__(self, model, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        
        # Handle DataParallel model for inference
        if isinstance(model, nn.DataParallel):
            self.model = model.module.to(device)
        else:
            self.model = model.to(device)
        
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 120, temperature: float = 0.8, 
                 top_k: int = 50, top_p: float = 0.9):
        """Generate text using nucleus (top-p) sampling"""
        
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
        
        generated = prompt_tokens.copy()
        
        for _ in range(max_length):
            if len(tokens[0]) >= self.model.max_seq_len:
                tokens = tokens[:, -self.model.max_seq_len+1:]
            
            logits, _ = self.model(tokens)
            logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            generated.append(next_token.item())
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated)
        return generated_text

