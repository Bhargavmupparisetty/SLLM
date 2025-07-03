import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sllm import SmallLLM, HuggingFaceBPETokenizer, TextGenerator

# CONFIG
CHECKPOINT_PATH = "/models/finetuned_model.pt"   
TOKENIZER_PATH = "/Tokenizer/bpe_tokenizer.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Tokenizer 
tokenizer = HuggingFaceBPETokenizer()
tokenizer.load(TOKENIZER_PATH)
vocab_size = tokenizer.vocab_size_actual

# Load Model
model = SmallLLM(
    vocab_size=vocab_size,
    dim=128,
    n_layers=6,
    n_heads=4,
    max_seq_len=128,
    dropout=0.1
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

# Generator Wrapper
generator = TextGenerator(model, tokenizer, DEVICE)

# Prompt Loop 
print("="*50)
print("TEXT GENERATION FROM TRAINED LLM")
print("Type 'exit' to quit.")
print("="*50)

while True:
    prompt = input("\nEnter your prompt: ")
    if prompt.strip().lower() == 'exit':
        break

    generated_text = generator.generate(
        prompt,
        max_length=120,
        temperature=0.9,
        top_k=50,
        top_p=0.9
    )

    print(f"\nGenerated:\n{generated_text}")
    print("-" * 50)
