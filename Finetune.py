import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')
 
from sllm import SmallLLM, HuggingFaceBPETokenizer, TextGenerator

class FineTuneDataset(Dataset):
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print("Processing texts for finetuning...")
        
        for text in tqdm(texts, desc="Tokenizing"):
            # Encode text
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            # Create sliding window chunks if text is longer than max_length
            if len(tokens) > max_length:
                for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.data.append(chunk)
            else:
                # Pad shorter sequences
                padded = tokens + [0] * (max_length - len(tokens))
                self.data.append(padded)
        
        print(f"Created {len(self.data)} training sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # Input is all tokens except the last one
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        
        # Target is all tokens except the first one
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

class ModelFineTuner:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            print(f"CUDA is available! Found {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA is not available. Using CPU.")
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = HuggingFaceBPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        print("Loading model...")
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.is_parallel = True
        else:
            self.is_parallel = False
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def load_model(self, model_path: str):

        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extracting model configuration from checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = SmallLLM(
                vocab_size=config['vocab_size'],
                dim=config['dim'],
                n_layers=config['n_layers'],
                n_heads=config['n_heads'],
                max_seq_len=config['max_seq_len'],
                dropout=config.get('dropout', 0.1)
            )
        else:
            # If information from model checkpoint not found
            model = SmallLLM(
                vocab_size=self.tokenizer.vocab_size_actual,
                dim=128,
                n_layers=6,
                n_heads=4,
                max_seq_len=128,
                dropout=0.1
            )
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def load_finetune_data(self, file_path: str) -> List[str]:

        print(f"Loading finetuning data from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
   
        if '\n\n' in content:
            # Splitting by double newlines 
            texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        elif '\n' in content:
            # Split by single newlines
            texts = [text.strip() for text in content.split('\n') if text.strip()]
        else:
            chunk_size = 500  
            texts = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # Filtering out very short texts
        texts = [text for text in texts if len(text) > 20]
        
        print(f"Loaded {len(texts)} text segments for finetuning")
        return texts
    
    def create_data_loaders(self, texts: List[str], batch_size: int = 8, 
                           val_split: float = 0.1, max_length: int = 128):
        
        # Splitting into train and validation
        val_size = int(len(texts) * val_split)
        train_texts = texts[val_size:]
        val_texts = texts[:val_size]
        
        print(f"Training texts: {len(train_texts)}, Validation texts: {len(val_texts)}")
        
        # Creating datasets
        train_dataset = FineTuneDataset(train_texts, self.tokenizer, max_length)
        val_dataset = FineTuneDataset(val_texts, self.tokenizer, max_length) if val_texts else None

        num_workers = min(4, os.cpu_count())
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True if num_workers > 0 else False
        ) if val_dataset else None
        
        return train_loader, val_loader
    
    def finetune(self, 
                 finetune_file: str,
                 num_epochs: int = 3,
                 learning_rate: float = 1e-5,
                 batch_size: int = 16, 
                 max_length: int = 128,
                 val_split: float = 0.1,
                 save_every: int = 1,
                 output_dir: str = './finetuned_model',
                 accumulation_steps: int = 1):  
    
        os.makedirs(output_dir, exist_ok=True)
        texts = self.load_finetune_data(finetune_file)
        train_loader, val_loader = self.create_data_loaders(
            texts, batch_size, val_split, max_length
        )
        
        effective_batch_size = batch_size * torch.cuda.device_count() * accumulation_steps
        print(f"Effective batch size: {effective_batch_size}")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * num_epochs // accumulation_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        print(f"Starting finetuning for {num_epochs} epochs...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Using {torch.cuda.device_count()} GPU(s)")
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
                input_ids = input_ids.to(self.device, non_blocking=True)
                target_ids = target_ids.to(self.device, non_blocking=True)
                
                logits, loss = self.model(input_ids, target_ids)
                
                if self.is_parallel:
                    loss = loss.mean()
                
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps  # Unscale for logging
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.4f}',
                    'Avg Loss': f'{train_loss/num_batches:.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}',
                    'GPU Mem': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
                })
                
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
            
            if len(train_loader) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for input_ids, target_ids in tqdm(val_loader, desc="Validation"):
                        input_ids = input_ids.to(self.device, non_blocking=True)
                        target_ids = target_ids.to(self.device, non_blocking=True)
                        
                        logits, loss = self.model(input_ids, target_ids)
                        
                        if self.is_parallel:
                            loss = loss.mean()
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
            
            # Saving finetuned model checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_model(epoch + 1, output_dir)
            
            # Clear cache after each epoch
            torch.cuda.empty_cache()
        
        # Saving final model
        self.save_model(num_epochs, output_dir, final=True)
        
        # Plotting training curves
        self.plot_training_curves(output_dir)
        
        print("Finetuning completed!")
        return self.model
    
    def save_model(self, epoch: int, output_dir: str, final: bool = False):

        model_to_save = self.model.module if self.is_parallel else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'model_config': {
                'vocab_size': model_to_save.embedding.num_embeddings,
                'dim': model_to_save.dim,
                'n_layers': len(model_to_save.layers),
                'n_heads': model_to_save.layers[0].attention.n_heads,
                'max_seq_len': model_to_save.max_seq_len,
                'dropout': 0.1
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if final:
            save_path = os.path.join(output_dir, 'finetuned_model_final.pt')
        else:
            save_path = os.path.join(output_dir, f'finetuned_model_epoch_{epoch}.pt')
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def plot_training_curves(self, output_dir: str):

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training Loss Over Time (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():

    MODEL_PATH = '/kaggle/input/sllm/pytorch/default/1/final_model_30k_steps (1).pt'
    TOKENIZER_PATH = '/kaggle/input/tokenizer-sllm/bpe_tokenizer.json'
    FINETUNE_FILE = '/kaggle/input/finetune/merged_output.txt'
    

    finetuner = ModelFineTuner(MODEL_PATH, TOKENIZER_PATH)
    
    finetuned_model = finetuner.finetune(
        finetune_file=FINETUNE_FILE,
        num_epochs=20,
        learning_rate=5e-6,
        batch_size=16,  
        max_length=128,
        val_split=0.1,
        save_every=1,
        output_dir='/kaggle/working/finetuned_model',
        accumulation_steps=2 
    )
    
    # Testing the finetuned model
    print("\n" + "="*50)
    print("Testing finetuned model:")
    print("="*50)
    
    model_for_inference = finetuned_model.module if isinstance(finetuned_model, nn.DataParallel) else finetuned_model
    
    generator = TextGenerator(model_for_inference, finetuner.tokenizer, finetuner.device)
    
    test_prompts = [
        "The future of AI",
        "In a world where"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        generated = generator.generate(prompt, max_length=100, temperature=0.8)
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main()
