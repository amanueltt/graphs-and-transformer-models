"""
Training script using CMA-ES evolutionary optimizer (no gradients)
Based on train_list_4_minimal.py but replaces AdamW with CMA-ES
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np
import torch
import cma

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.updated_model_5 import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Arguments
parser = argparse.ArgumentParser(description='CMA-ES training for transformer models')
parser.add_argument('--dataset', type=str, default='list')
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--n_embd', type=int, default=120)
parser.add_argument('--max_iters', type=int, default=500, help='Number of CMA-ES generations')
parser.add_argument('--min_value', type=int, default=0)
parser.add_argument('--max_value', type=int, default=100)
parser.add_argument('--is_sorted', type=str, default="True")
parser.add_argument('--num_list_copies', type=int, default=5)
parser.add_argument('--use_identity_embeddings', type=bool, default=False)
parser.add_argument('--use_fixed_positions', type=bool, default=False)
parser.add_argument('--use_identity_V', type=bool, default=False)
parser.add_argument('--fixed_length', type=int, default=None)
parser.add_argument('--permutation_type', type=str, default="reversal")
parser.add_argument('--eval_batch_size', type=int, default=256, help='Batch size for fitness evaluation')
parser.add_argument('--sigma', type=float, default=0.5, help='Initial CMA-ES step size')
parser.add_argument('--popsize', type=int, default=20, help='CMA-ES population size')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup paths
list_type = "sorted" if args.is_sorted == "True" else "unsorted"
length_type = f"fixed{args.fixed_length}" if args.fixed_length is not None else "variable"
data_dir = os.path.join('data', f'{args.dataset}/{list_type}/{length_type}/{args.min_value}-{args.max_value}/{args.permutation_type}')

# Load metadata
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
block_size = meta['block_size']
vocab_size = meta['vocab_size']

# Output directory
embedding_suffix = "_cmaes"  # Mark this as CMA-ES trained
config = f"{args.n_layer}_{args.n_head}_{args.n_embd}{embedding_suffix}"
out_dir = f'out/{args.dataset}_{list_type}_{length_type}_{args.permutation_type}_{config}_{args.min_value}-{args.max_value}'
os.makedirs(out_dir, exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Data loading
if args.num_list_copies == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{args.num_list_copies}.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split, batch_size):
    """Get a batch of data"""
    data = train_data if split == 'train' else val_data
    data_size = block_size + 1
    max_idx = (len(data) - data_size) // data_size
    ix = torch.randint(max_idx, (batch_size,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# Model initialization
model_args = dict(
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    block_size=block_size,
    bias=False,
    vocab_size=vocab_size,
    dropout=0.0,  # No dropout for CMA-ES
    use_identity_embeddings=args.use_identity_embeddings,
    use_fixed_positions=args.use_fixed_positions,
    use_identity_V=args.use_identity_V
)

print("Initializing model...")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
model.eval()  # Always in eval mode for CMA-ES

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

# -----------------------------------------------------------------------------
# CMA-ES helper functions

def get_flat_params(model):
    """Extract all trainable parameters as a flat numpy array"""
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p.data.cpu().numpy().flatten())
    return np.concatenate(params)

def set_flat_params(model, flat_params):
    """Load flat numpy array back into model parameters"""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                size = p.numel()
                p.data = torch.tensor(
                    flat_params[idx:idx+size], 
                    dtype=p.dtype, 
                    device=p.device
                ).reshape(p.shape)
                idx += size

@torch.no_grad()
def evaluate_fitness(model, num_batches=5):
    """Evaluate model fitness (lower is better = loss)"""
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch('train', args.eval_batch_size)
        _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / num_batches

@torch.no_grad()
def estimate_val_loss(model, num_batches=10):
    """Estimate validation loss"""
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch('val', args.eval_batch_size)
        _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / num_batches

# -----------------------------------------------------------------------------
# Initialize CMA-ES

initial_params = get_flat_params(model)
print(f"Initial params shape: {initial_params.shape}")
print(f"Sigma (initial step size): {args.sigma}")
print(f"Population size: {args.popsize}")

es = cma.CMAEvolutionStrategy(
    initial_params,
    args.sigma,
    {'popsize': args.popsize, 'maxiter': args.max_iters}
)

# -----------------------------------------------------------------------------
# Training loop

print(f"\nStarting CMA-ES optimization...")
print(f"Max generations: {args.max_iters}")
print("-" * 50)

best_val_loss = float('inf')
generation = 0
log_file = os.path.join(out_dir, 'cmaes_training.log')

with open(log_file, 'w') as f:
    f.write(f"CMA-ES Training Log\n")
    f.write(f"Parameters: {num_params}\n")
    f.write(f"Sigma: {args.sigma}, Popsize: {args.popsize}\n\n")

t0 = time.time()

while not es.stop():
    generation += 1
    
    # Get candidate solutions
    solutions = es.ask()
    
    # Evaluate each candidate
    fitnesses = []
    for solution in solutions:
        set_flat_params(model, solution)
        fitness = evaluate_fitness(model, num_batches=3)
        fitnesses.append(fitness)
    
    # Update CMA-ES
    es.tell(solutions, fitnesses)
    
    # Get best solution and set model to it
    best_idx = np.argmin(fitnesses)
    set_flat_params(model, solutions[best_idx])
    
    # Logging every 10 generations
    if generation % 10 == 0 or generation == 1:
        val_loss = estimate_val_loss(model)
        train_loss = min(fitnesses)
        elapsed = time.time() - t0
        
        log_msg = f"Gen {generation}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={elapsed:.1f}s"
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model': model.state_dict(),
                'model_args': model_args,
                'generation': generation,
                'best_val_loss': best_val_loss,
                'cmaes_sigma': es.sigma,
            }
            ckpt_path = os.path.join(out_dir, f'{generation}_ckpt_{args.num_list_copies}.pt')
            torch.save(checkpoint, ckpt_path)
            print(f"        Saved checkpoint: {ckpt_path}")

# -----------------------------------------------------------------------------
# Final save

print("\n" + "=" * 50)
print("CMA-ES optimization complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Total generations: {generation}")

# Save final model with generation number
final_checkpoint = {
    'model': model.state_dict(),
    'model_args': model_args,
    'generation': generation,
    'best_val_loss': best_val_loss,
}
final_path = os.path.join(out_dir, f'{generation}_ckpt_{args.num_list_copies}.pt')
torch.save(final_checkpoint, final_path)
print(f"Final model saved to: {final_path}")
