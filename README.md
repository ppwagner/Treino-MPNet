# Bayesian Attention Mechanism
This repository contains the implementation of the Bayesian Attention Mechanism (BAM) as described in the paper "Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation" by [Author(s)](link_to_paper). The training code was adapted from [llm.c](https://github.com/karpathy/llm.c), and the model codes used [llama 3](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py) as a reference.

# Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
# Usage
To train the BAM model, first prepare your dataset with the following command:

```bash
python dataset.py
```

This should create a tokenized dataset file in the `data/`, of the [FineWeb 10B token sample](https://huggingface.co/datasets/HuggingFaceFW/fineweb) using [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) tokenizer. If you want to use larger datasets `--streaming` option can be used to stream the dataset from Hugging Face.
To train the BAM model as described in the paper, run the following command adapting to your hardware configuration: 

```bash
time torchrun --standalone --nproc_per_node '<customize>' \
    train.py \
        --tokens_per_step=589824 \
        --position_encoding=bam_ssmax \
        --model_size=l12 \
        --sequence_length=512 \
        --batch_size='<customize>' \
        --weight_decay=0.1  \
        --learning_rate_decay_frac=0.1 \
        --compile \
        --tensorcores \
        --val_loss_every=32 \
        --dtype=bfloat16 \
        --learning_rate 1e-3 \
```

# Evaluation
To evaluate the BAM model, you can use the following command:

```bash
python evaluate.py --log_dir '<path_to_your_model_log_dir>' 
```
Considering a single run of the training example above, the log directory would be `logs/l12/bam_ssmax/version_00`.



# Implementation Details

The Bayesian Attention Mechanism (BAM) model is implemented in the `models/bam.py` and `models/bam_ssmax.py` files. The models use the following class implementations to generate biases for the attention mechanism:

```python

class AttentionPrior(nn.Module):
    def __init__(self, args: SSMaxBATModelArgs):
        super().__init__()
        self.seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.eps = 1e-5

        
        if args.theta_alpha_init == 'slope':
            theta_alpha = torch.tensor(get_slopes(args.n_heads), dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.theta_alpha_init == 'sampled':
            theta_alpha = torch.randn((1, args.n_heads, 1, 1), dtype=torch.float).exp()
        else:
            theta_alpha = torch.full((1, args.n_heads, 1, 1), float(args.theta_alpha_init), dtype=torch.float)
        
        if args.train_theta_beta and args.thata_beta_init == 'linear':
            theta_beta  = torch.linspace(0, 1, args.n_heads, dtype=torch.float).reshape(1, args.n_heads, 1, 1)
        elif args.train_theta_beta and args.thata_beta_init == 'sampled':
            theta_beta  = torch.randn((1, args.n_heads, 1, 1), dtype=torch.float)
        elif args.train_theta_beta:
            theta_beta   = torch.full((1, args.n_heads, 1, 1), float(args.thata_beta_init), dtype=torch.float)
        else:
            theta_beta   = torch.ones((1, args.n_heads, 1, 1), dtype=torch.float)

        theta_mu = torch.full((1, args.n_heads, 1, 1), float(args.theta_mu_init),   dtype=torch.float)
        
        self.theta_beta  = nn.Parameter(theta_beta, requires_grad = args.train_theta_beta)
        self.theta_alpha = nn.Parameter(theta_alpha, requires_grad = args.train_theta_alpha)
        self.theta_mu    = nn.Parameter(theta_mu,   requires_grad = args.train_theta_mu)

    def forward(self, seq_len=None):
        seq_len = seq_len or self.seq_len
        positions = torch.arange(seq_len, device=self.theta_alpha.device).float()
        b = (positions[None, :] - positions[:, None]).reshape(1, 1, seq_len, seq_len)
        b = b - (self.theta_mu.exp() - (-self.theta_mu).exp())
        return -((b.abs() + self.eps) ** self.theta_beta) * self.theta_alpha.exp() 
    
```