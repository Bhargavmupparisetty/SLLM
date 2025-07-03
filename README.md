# SLLM (Small language Model) - A Lightweight Transformer-based Language Model


**SLLM**, is a compact, efficient sub-word Language Model trained from scratch and fine-tuned for specialized tasks. It's designed for fast experimentation and deep understanding of LLM internals.

---

## Pretraining Details

The model was pretrained on a clean English corpus that is extracted from Wikipedia using Wiki_crawler. The dataset contains general knowledge concepts taken from wikipedia about MAthematics, Quantum physics, Sports and others. The pre-training phase details include:

- **Vocabulary**: 4000 tokens (HuggingFace BPE)
- **Training steps**: 50,000
- **Sequence length**: 128
- **Architecture**: 6 Transformer blocks, 4 heads, 128 hidden dim
- **Objective**: Next-token prediction using cross-entropy loss
- **Parameters**: 2.2M Parameters

###  Pretraining Metrics

| ![Pretraining Metrics](metrics/pretraining_loss.png) |

---

##  Fine-Tuning Details

The pretrained model was then **fine-tuned on task-specific text data**, on story texts collected.

- **Epochs**: 20
- **Tokenizer reused**: HuggingFace BPE trained from pretraining
- **Learning rate**: Warmup + cosine decay

###  Fine-Tuning Metrics


| ![Fine-tuning Metrics](metrics/finetuning_loss.png) | 

---

##  Features

-  BPE Tokenizer from HuggingFace
-  Rotary Positional Embeddings (RoPE)
-  SwiGLU Activation Function
-  RMSNorm instead of LayerNorm
-  Causal Masking for autoregressive modeling
-  Text Generation with nucleus sampling (top-k + top-p)

---

##  Try Text Generation

```bash
python Test.py
