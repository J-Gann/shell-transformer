# Shell Transformer: Bash Command Generation with Transformers

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Jonas Gann

**Course:** Generative Neural Networks, University of Heidelberg

## 🎯 Project Overview

This project implements a Transformer model from scratch to generate bash commands, serving as both an educational tool for understanding Transformer architecture and a practical bash command autocompletion system. Instead of training on natural language, we focus on shell command patterns to create a specialized generative model for command-line interactions.

## ✨ Features

- **Custom Transformer Implementation**: Built from scratch with multi-head self-attention
- **Bash Command Generation**: Trained specifically on shell command datasets
- **Character-level Tokenization**: 543-token vocabulary for shell commands
- **Hyperparameter Optimization**: Integrated Optuna for automated tuning
- **Experiment Tracking**: Weights & Biases (wandb) integration
- **Interactive Notebooks**: Easy-to-use Jupyter interfaces for training and inference

## 🏗️ Architecture

The implementation includes several key components:

### Core Components
- **SelfAttentionBlock**: Single attention head with causal masking
- **MultiHeadSelfAttention**: Multiple parallel attention heads with projection
- **TransformerDecoder**: Stack of transformer decoder layers with residual connections
- **Positional Encoding**: Learned positional embeddings for sequence modeling

### Model Configuration
- **Block Size**: 256 tokens (configurable)
- **Vocabulary**: 543 unique characters from shell commands
- **Dropout**: 0.2 for regularization
- **Architecture**: Decoder-only transformer (GPT-style)

## 📁 Project Structure

```
shell-transformer/
├── README.md                           # This file
├── transformer.ipynb                   # Main training and inference notebook
├── data.ipynb                         # Data preprocessing and analysis
├── stoi                               # String-to-index vocabulary mapping
├── itos                               # Index-to-string vocabulary mapping
├── optuna.db                          # Hyperparameter optimization database
└── final_with_preprocessing/
    └── jumping-river-27/
        └── shell_transformer_23000    # Trained model checkpoint
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch numpy optuna wandb plotly bashlex
```

### Using the Pre-trained Model

1. **Load and Generate Commands**:
   ```python
   # Open transformer.ipynb and run the cells to:
   # - Load the pre-trained model
   # - Generate new bash commands
   # - Experiment with different prompts
   ```

2. **Interactive Generation**:
   The notebook provides an easy interface to:
   - Input partial commands
   - Generate completions
   - Explore model predictions

### Training from Scratch

1. **Data Preparation**:
   ```python
   # Use data.ipynb to:
   # - Load bash command datasets
   # - Clean and preprocess data
   # - Create vocabulary mappings
   ```

2. **Model Training**:
   ```python
   # In transformer.ipynb:
   # - Configure hyperparameters
   # - Train the model
   # - Monitor training with wandb
   # - Save checkpoints
   ```

## 📊 Dataset

The model is trained on multiple bash command datasets:

1. **MUNI KYPO Commands**: Shell commands from cybersecurity training environments
2. **Bash History Dataset**: Real-world bash command histories
3. **Shell Dataset**: Curated shell command examples

**Total Commands**: ~100k+ bash commands  
**Vocabulary Size**: 543 unique characters  
**Command Types**: File operations, system commands, git operations, package management, etc.

## 🛠️ Model Details

### Hyperparameters
- **Embedding Size**: Configurable (typically 128-512)
- **Number of Layers**: Optimized via Optuna
- **Attention Heads**: Configurable multi-head setup
- **Learning Rate**: Adaptive with evaluation-based scheduling
- **Batch Size**: Optimized for available hardware

### Training Process
- **Loss Function**: Cross-entropy loss for next-token prediction
- **Optimization**: Adam optimizer with learning rate scheduling
- **Evaluation**: Regular validation on held-out test set
- **Early Stopping**: Based on validation loss improvements

## 📈 Evaluation

The model performance is evaluated on:
- **Perplexity**: Measure of prediction uncertainty
- **Generation Quality**: Manual assessment of generated commands
- **Completion Accuracy**: How well it completes partial commands
- **Syntax Validity**: Whether generated commands are syntactically correct

## 🔧 Configuration

Key configuration parameters in the notebooks:

```python
# Model Configuration
block_size = 256        # Maximum sequence length
dropout = 0.2          # Dropout rate
eval_interval = 500    # Evaluation frequency
eval_iters = 200       # Evaluation iterations

# Training Configuration
batch_size = 64        # Training batch size
learning_rate = 1e-3   # Initial learning rate
max_iters = 10000      # Maximum training iterations
```

## 🎯 Usage Examples

### Command Completion
```bash
Input:  "git add"
Output: "git add ."
        "git add -A"
        "git add file.py"
```

### System Commands
```bash
Input:  "ls -"
Output: "ls -la"
        "ls -lah"
        "ls -lt"
```

### File Operations
```bash
Input:  "cp "
Output: "cp file.txt backup/"
        "cp -r directory/ destination/"
```

## 🔍 Future Improvements

- **Context Awareness**: Incorporate current directory and file listings
- **Command Validation**: Add syntax checking for generated commands
- **Interactive CLI**: Build a command-line interface for real-time completion
- **Fine-tuning**: Domain-specific adaptation for different environments
- **Multi-modal**: Incorporate command documentation and man pages

## 📚 Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Optuna**: Hyperparameter optimization
- **Weights & Biases**: Experiment tracking
- **bashlex**: Bash command parsing
- **Plotly**: Interactive visualizations

### Hardware Requirements
- **GPU**: Recommended for training (CUDA support)
- **RAM**: 8GB+ for training, 4GB+ for inference
- **Storage**: 1GB+ for datasets and model checkpoints

## 🤝 Contributing

This is an educational project for the Generative Neural Networks course. If you'd like to extend or improve the model:

1. Fork the repository
2. Create a feature branch
3. Implement your improvements
4. Add tests and documentation
5. Submit a pull request

## 🏫 Academic Context

This project was developed as part of the "Generative Neural Networks" course at the University of Heidelberg. The goal was to implement a Transformer model from scratch to gain hands-on experience with:

- Attention mechanisms
- Transformer architecture
- Autoregressive generation
- Sequence modeling
- Neural language modeling

## 📞 Contact

- **Jonas Gann**: [GitHub Profile](https://github.com/J-Gann)

---

*Built with ❤️ for learning and understanding Transformer architectures*
