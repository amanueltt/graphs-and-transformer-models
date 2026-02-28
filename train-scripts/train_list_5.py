"""
Training script for no-MLP model with full-dim + sum attention (updated_model_5)
"""
import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import re
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.updated_model_5 import GPTConfig, GPT  # Updated import
from logger import get_logger
import logging

# -----------------------------------------------------------------------------
# the input parameters
parser = argparse.ArgumentParser(description='Training of the NanoGPT without MLP layers (full-dim + sum attention).')
parser.add_argument('--dataset', type=str, default='list', help='Name of the dataset to use')  
parser.add_argument('--n_layer', type=int, default=1, help='Number of layers (default: 1)')  
parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads (default: 1)')  
parser.add_argument('--n_embd', type=int, default=120, help='Size of the embeddings (default: 120)')
parser.add_argument('--max_iters', type=int, default=10000, help='Number of Iterations (default: 10000)')
parser.add_argument('--min_value', type=int, default=0, help='Min value in lists')
parser.add_argument('--max_value', type=int, default=100, help='Max value in lists')
parser.add_argument('--is_sorted', type=str, default="True", help='Whether lists are sorted')
parser.add_argument('--num_list_copies', type=int, default=5, help='Number of copies per list')
parser.add_argument('--use_identity_embeddings', type=bool, default=False, help='Use identity matrix for embeddings (default: False)')
parser.add_argument('--use_fixed_positions', type=bool, default=False, help='Use fixed positional embeddings (default: False)')
parser.add_argument('--use_identity_V', type=bool, default=False, help='Use identity matrix for V projection (default: False)')
parser.add_argument('--fixed_length', type=int, default=None, help='Fixed length of lists if specified')
parser.add_argument('--permutation_type', type=str, default="reversal", help='Type of permutation (default: reversal)')
parser.add_argument('--train_batch_size', type=int, default=256, help='Training batch size (default: 256)')
parser.add_argument('--learning_rate', type=float, default=5e-2, help='Max learning rate (default: 5e-2)')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (default: 0.0)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
args = parser.parse_args()

dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
min_value = args.min_value
max_value = args.max_value
is_sorted = args.is_sorted
num_list_copies = args.num_list_copies
use_identity_embeddings = args.use_identity_embeddings
use_fixed_positions = args.use_fixed_positions
use_identity_V = args.use_identity_V
fixed_length = args.fixed_length
permutation_type = args.permutation_type

# Determine list type directory
list_type = "sorted" if is_sorted == "True" else "unsorted"
length_type = f"fixed{fixed_length}" if fixed_length is not None else "variable"
data_dir = os.path.join('data', f'{dataset}/{list_type}/{length_type}/{min_value}-{max_value}/{permutation_type}')

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
    
stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']

# Validate n_embd vs block_size for fixed positions
if use_fixed_positions and n_embd <= block_size:
    raise ValueError(f"When using fixed positions, n_embd ({n_embd}) must be larger than block_size ({block_size}). "
                     f"Suggestion: use n_embd >= {block_size + 32}")


# Build config suffix (no W_O / c_proj in this model, reflect that in the name)
embedding_suffix = ""

if use_identity_embeddings and use_fixed_positions and use_identity_V:
    embedding_suffix = "_identityE_fixedP_identityV"
elif use_identity_embeddings and use_fixed_positions and not use_identity_V:
    embedding_suffix = "_identityE_fixedP"
elif use_identity_embeddings and not use_fixed_positions and use_identity_V:
    embedding_suffix = "_identityE_identityV"
elif use_identity_embeddings and not use_fixed_positions and not use_identity_V:
    embedding_suffix = "_identityE"
elif not use_identity_embeddings and use_fixed_positions and use_identity_V:
    embedding_suffix = "_fixedP_identityV"
elif not use_identity_embeddings and use_fixed_positions and not use_identity_V:
    embedding_suffix = "_fixedP"
elif not use_identity_embeddings and not use_fixed_positions and use_identity_V:
    embedding_suffix = "_identityV"
else:
    embedding_suffix = ""

# Add suffix
no_mlp_suffix = "_no_mlp"
embedding_suffix += no_mlp_suffix

log_suffix = embedding_suffix

# Modify the config to include embedding settings
config = f"{n_layer}_{n_head}_{n_embd}{embedding_suffix}"
out_dir = f'out/{dataset}_{list_type}_{length_type}_{permutation_type}_{config}_{min_value}-{max_value}'

# -----------------------------------------------------------------------------
# default config values
# I/O
eval_interval = max_iters // 10
log_interval = max_iters // 100
eval_iters = max_iters // 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
gradient_accumulation_steps = 1
print(f"Using Gradient Accumulation Steps: {gradient_accumulation_steps}")
train_batch_size = args.train_batch_size
print(f"Using Training Batch Size: {train_batch_size}")
val_batch_size = args.train_batch_size // 2
batch_size = train_batch_size
bias = False
# adamw optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = max_iters//20
min_lr = args.learning_rate/10
lr_decay_iters = max_iters
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16'
compile = True
# -----------------------------------------------------------------------------
# Use values from args
learning_rate = args.learning_rate
dropout = args.dropout
weight_decay = args.weight_decay
print(f"Using regularization with learning_rate={learning_rate}, warmup_iters={warmup_iters}, dropout={dropout}, weight_decay={weight_decay}")
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
if(num_list_copies == 0):
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_list_copies}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Calculate epoch info for tracking
total_sequences = len(train_data) // (block_size + 1)
iterations_per_epoch = total_sequences // train_batch_size
print(f"Total training sequences: {total_sequences}")
print(f"Iterations per epoch: {iterations_per_epoch}")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size
    data_size = block_size + 1
    ix = torch.randint( (len(data) - data_size)//data_size , (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# logger
if(num_list_copies == 0):
    logger = get_logger(os.path.join(out_dir, f"no_output_train{log_suffix}.log"))
    log_file_name = os.path.join(out_dir, f"train{log_suffix}.log")
else:
    logger = get_logger(os.path.join(out_dir, f'no_output_train_{num_list_copies}{log_suffix}.log'))
    log_file_name = os.path.join(out_dir, f"train_{num_list_copies}{log_suffix}.log")

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
stoi, itos = meta['stoi'], meta['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# model init — note: no use_identity_output_projection for model 5
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    vocab_size=None, 
    dropout=dropout,
    use_identity_embeddings=use_identity_embeddings,
    use_fixed_positions=use_fixed_positions,
    use_identity_V=use_identity_V,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch (NO MLP, full-dim + sum attention)")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'use_identity_embeddings', 'use_fixed_positions', 'use_identity_V']:
        if k in ['use_identity_embeddings', 'use_fixed_positions', 'use_identity_V'] and k not in checkpoint_model_args:
            continue
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# Log the configuration
print(f"Using identity embeddings: {model_args.get('use_identity_embeddings', False)}")
print(f"Using fixed positions: {model_args.get('use_fixed_positions', False)}")
print(f"Using identity V: {model_args.get('use_identity_V', False)}")
print(f"Model architecture: NO MLP, full-dim + sum attention (no W_O projection)")

print(f"Block size (context window): {block_size}")

if use_fixed_positions:
    token_emb_dim = n_embd - block_size
    pos_emb_dim = block_size
    print(f"Embedding breakdown: {token_emb_dim} token dims + {pos_emb_dim} positional dims = {n_embd} total")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# initialize a GradScaler
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def open_and_append(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
accuracy = []
corrects = []
totals = []

while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        current_epoch = iter_num // iterations_per_epoch if iterations_per_epoch > 0 else 0
        print(f"step {iter_num} (epoch {current_epoch}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num} (epoch {current_epoch}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        open_and_append(log_file_name, f"step {iter_num} (epoch {current_epoch}): train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "epoch": current_epoch,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                logger.info(f"saving checkpoint to {out_dir}")
                open_and_append(log_file_name, f"saving checkpoint to {out_dir}")
                
                if num_list_copies == 0:
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))
                else:
                    torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt_{num_list_copies}.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        current_epoch = iter_num // iterations_per_epoch if iterations_per_epoch > 0 else 0
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num} (epoch {current_epoch}): loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
        logger.info(f"iter {iter_num} (epoch {current_epoch}): loss {lossf:.4f}")
        open_and_append(log_file_name, f"iter {iter_num} (epoch {current_epoch}): loss {lossf:.4f}")
    
    iter_num += 1
    local_iter_num += 1
    
    if iter_num > max_iters:
        break

torch.save(torch.tensor(corrects).cpu(), os.path.join(out_dir, f'corrects{log_suffix}.pt'))
torch.save(torch.tensor(totals).cpu(), os.path.join(out_dir, f'totals{log_suffix}.pt'))

if ddp:
    destroy_process_group()
