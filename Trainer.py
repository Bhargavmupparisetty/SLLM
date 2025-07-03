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
 
from sllm import SmallLLM, HuggingFaceBPETokenizer, TextGenerator, TextDataset






class Trainer:
    def __init__(self, model, train_loader, device, lr=1e-3, max_steps=30000):
        self.device = device
        self.train_loader = train_loader
        self.max_steps = max_steps
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(model)
            self.model = self.model.to(device)
            self.is_parallel = True
        else:
            self.model = model.to(device)
            self.is_parallel = False
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Cosine learning rate schedule with warmup
        self.warmup_steps = 1000  
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=lr,
            total_steps=self.max_steps,
            pct_start=self.warmup_steps / self.max_steps,  
            anneal_strategy='cos'
        )
        
        # Training metrics
        self.train_losses = []
        self.learning_rates = []
        self.step_count = 0

    def train(self):
        print(f"Starting training for {self.max_steps} steps...")
        
        self.model.train()
        
        data_iter = iter(self.train_loader)
        
        running_loss = 0.0
        log_interval = 100 
        save_interval = 5000  # Save checkpoint every 5000 steps
        
        for step in range(self.max_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)
            
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            
            # Handle DataParallel loss averaging
            if self.is_parallel:
                loss = loss.mean()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            self.step_count += 1
            
            # Log progress
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Step {step + 1}/{self.max_steps}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
                
                # Store metrics
                self.train_losses.append(avg_loss)
                self.learning_rates.append(current_lr)
                
                running_loss = 0.0
            
            # Save checkpoint
            if (step + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{step + 1}.pt")
            
            if (step + 1) % 1000 == 0:
                torch.cuda.empty_cache()
        
        print(f"Training completed! Total steps: {self.max_steps}")
        
        # Save final checkpoint
        self.save_checkpoint("final_model_30k_steps.pt")

    def save_checkpoint(self, filename: str):
        model_state = self.model.module.state_dict() if self.is_parallel else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
            'step_count': self.step_count,
            'max_steps': self.max_steps
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def plot_metrics(self):
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training loss
        steps = [i * 100 for i in range(len(self.train_losses))]  
        ax1.plot(steps, self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(steps, self.learning_rates)
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics_30k_steps.png', dpi=300, bbox_inches='tight')
        plt.show()
