"""
Attention Pattern Visualization Script for updated_model_5 (full-dim + sum)
"""
import os
import sys
import math
import argparse
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.updated_model_5 import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize attention patterns for model 5')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to meta.pkl file')
    parser.add_argument('--input', type=str, default=None,
                        help='Single input sequence (space-separated tokens)')
    parser.add_argument('--inputs_file', type=str, default=None,
                        help='Path to test file with multiple inputs (one per line)')
    parser.add_argument('--custom_inputs', type=str, nargs='+', default=None,
                        help='List of custom input sequences to average over')
    parser.add_argument('--target_length', type=int, default=None,
                        help='Filter inputs to this sequence length (for averaging)')
    parser.add_argument('--heads', type=int, nargs='+', default=[0],
                        help='Attention heads to visualize (default: 0)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0],
                        help='Transformer layers to visualize (default: 0)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization (default: show inline)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (default: cpu)')
    return parser.parse_args()


class ModelWithAttention(GPT):
    """Wrapper class that extends GPT (model 5) to return attention weights"""
    
    def forward(self, idx, targets=None, return_attn_weights=False):
        device = idx.device
        b, t = idx.size()
        
        # Get embeddings
        tok_emb = self.transformer.wte(idx)
        
        if self.config.use_fixed_positions:
            if self.config.use_identity_embeddings:
                x = tok_emb
            else:
                pos_emb = self.transformer.wpe(t)
                pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)
                x = torch.cat([tok_emb, pos_emb], dim=-1)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        
        x = self.transformer.drop(x)
        
        attention_weights = []
        
        if return_attn_weights:
            for block in self.transformer.h:
                normalized = block.ln_1(x)
                
                # Model 5 uses a ModuleList of QKV projections, one for each head
                # We need to collect attention weights from each head
                layer_attns = []
                
                for h_idx, qkv_proj in enumerate(block.attn.head_qkv):
                    qkv = qkv_proj(normalized)
                    q, k, v = qkv.split(block.attn.n_embd, dim=2)
                    
                    B, T, C = q.size()
                    
                    # Calculate attention scores
                    scale = 1.0 / math.sqrt(C)
                    att = (q @ k.transpose(-2, -1)) * scale   # (B, T, T)
                    
                    # Apply causal mask
                    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                    att = att.masked_fill(mask.unsqueeze(0), float('-inf'))
                    
                    # Softmax
                    att = torch.nn.functional.softmax(att, dim=-1)
                    
                    layer_attns.append(att.detach())
                
                # layer_attns is a list of (B, T, T) tensors, one per head
                # Stack them along head dimension to match model_4 format: (B, n_head, T, T)
                stacked_layer_attn = torch.stack(layer_attns, dim=1)
                attention_weights.append(stacked_layer_attn)
                
                # Continue forward pass
                # Accumulate the actual output
                out = torch.zeros_like(x)
                for h_idx, qkv_proj in enumerate(block.attn.head_qkv):
                    qkv = qkv_proj(normalized)
                    q, k, v = qkv.split(block.attn.n_embd, dim=2)
                    if block.attn.use_identity_V: v = normalized
                    
                    if block.attn.flash:
                        q = q.unsqueeze(1)
                        k = k.unsqueeze(1)
                        v = v.unsqueeze(1)
                        y = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=None,
                            dropout_p=block.attn.dropout if block.attn.training else 0,
                            is_causal=True
                        )
                        y = y.squeeze(1)
                    else:
                        scale = 1.0 / math.sqrt(C)
                        att_fwd = (q @ k.transpose(-2, -1)) * scale
                        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
                        att_fwd = att_fwd.masked_fill(mask.unsqueeze(0), float('-inf'))
                        att_fwd = torch.nn.functional.softmax(att_fwd, dim=-1)
                        att_fwd = block.attn.attn_dropout(att_fwd)
                        y = att_fwd @ v
                    
                    out = out + y
                
                out = out / block.attn.n_head
                out = block.attn.resid_dropout(out)
                
                x = x + out
            
            x = self.transformer.ln_f(x)
            
            if targets is not None:
                logits = self.lm_head(x)
                import torch.nn.functional as F
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
            
            return logits, loss, attention_weights
        else:
            for block in self.transformer.h:
                x = x + block.attn(block.ln_1(x))
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, [-1], :]) if targets is None else self.lm_head(x)
            loss = None
            return logits, loss


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = ModelWithAttention(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, gptconf


def visualize_attention(model, input_tokens, token_labels, heads, layers, config, save_path=None, show_average=False, focus_q_token="?", scale=1.0, max_cols=4):
    """Visualize attention patterns"""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        _, _, attention_weights = model(input_tensor, return_attn_weights=True)
    
    import math
    plot_items = []
    
    for layer_idx in layers:
        if layer_idx >= len(attention_weights):
            continue
            
        layer_weights = attention_weights[layer_idx]
        
        for head_idx in heads:
            if head_idx >= layer_weights.shape[1]:
                continue
            attn_matrix = layer_weights[0, head_idx].cpu().numpy()
            title = f'Layer {layer_idx}, Head {head_idx}'
            plot_items.append((attn_matrix, title))
            
        if show_average:
            avg_matrix = layer_weights[0, heads].mean(dim=0).cpu().numpy()
            title = f'Layer {layer_idx}, Average of Heads'
            plot_items.append((avg_matrix, title))
            
    total_plots = len(plot_items)
    if total_plots == 0:
        print("No valid attention heads to visualize.")
        return
        
    n_cols = min(max_cols, total_plots)
    n_rows = math.ceil(total_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols*scale, 5*n_rows*scale), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (attn_matrix, title) in enumerate(plot_items):
        ax = axes_flat[idx]
        sns.heatmap(attn_matrix, cmap="viridis", cbar=True, square=True, 
                   annot=False, ax=ax, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold' if 'Average' in title else 'normal')
        
        ax.set_xticks(np.arange(len(token_labels)) + 0.5)
        ax.set_yticks(np.arange(len(token_labels)) + 0.5)
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(token_labels, fontsize=9)
        ax.set_xlabel('Key (attending to)', fontsize=10)
        ax.set_ylabel('Query', fontsize=10)
        
        if focus_q_token in token_labels:
            try:
                q_idx = token_labels.index(focus_q_token)
                if q_idx > 0:
                    import matplotlib.patches as patches
                    rect = patches.Rectangle((0, 0), len(token_labels), q_idx, 
                                             linewidth=0, edgecolor='none', facecolor='white', alpha=0.6)
                    ax.add_patch(rect)
                
                # Mark the highest attended token between the first two tokens (u and v)
                if len(token_labels) > 1:
                    c_indices = [0, 1]
                    vals = [attn_matrix[q_idx, c] for c in c_indices]
                    highest_c_idx = 0 if vals[0] >= vals[1] else 1
                    ax.text(highest_c_idx + 0.5, q_idx + 0.5, "*", ha="center", va="center", 
                            color="white", fontweight="bold", fontsize=20)
            except ValueError:
                pass
                
    # Hide any unused subplots
    for idx in range(total_plots, n_rows * n_cols):
        axes_flat[idx].axis('off')
        
    plt.suptitle(f'Attention Patterns\nInput: {" ".join(token_labels)}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_attention_batch(model, input_text_block, stoi, heads, layers, config, 
                               save_path=None, scale=1.0, max_cols=2, 
                               focus_q_token="?"):
    """
    Visualize average attention for multiple sequences.
    
    Args:
        model: The trained model
        input_text_block: Multi-line string where each line is a sequence like "u v h t 0 ? u"
                          The last token is the correct answer.
                          Model input is everything up to and including '?'.
        stoi: String-to-index vocabulary mapping
        heads: List of head indices to average over
        layers: List of layer indices to visualize
        config: Model config
        save_path: Optional path to save the figure
        scale: Plot scaling factor
        max_cols: Maximum number of columns in the grid
        focus_q_token: The query token to focus on (default '?')
    """
    import math
    import matplotlib.patches as mpatches
    
    device = next(model.parameters()).device
    
    # Parse input lines
    lines = [line.strip() for line in input_text_block.strip().split('\n') if line.strip()]
    
    plot_items = []  # Each item: (avg_attn_matrix, input_labels, correct_answer, title)
    
    for line in lines:
        tokens = line.split()
        
        # Last token is the correct answer
        correct_answer = tokens[-1]
        
        # Model input is everything up to and including '?'
        if focus_q_token in tokens:
            q_pos = tokens.index(focus_q_token)
            input_tokens_str = tokens[:q_pos + 1]  # up to and including '?'
        else:
            print(f"Warning: '{focus_q_token}' not found in '{line}', skipping")
            continue
        
        # Encode tokens
        try:
            input_ids = [stoi[t] for t in input_tokens_str]
        except KeyError as e:
            print(f"Skipping '{line}': token {e} not in vocabulary")
            continue
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            _, _, attention_weights = model(input_tensor, return_attn_weights=True)
        
        # Compute average attention across heads for each layer
        for layer_idx in layers:
            if layer_idx >= len(attention_weights):
                continue
            layer_weights = attention_weights[layer_idx]
            
            # Average across all specified heads
            valid_heads = [h for h in heads if h < layer_weights.shape[1]]
            if not valid_heads:
                continue
            avg_matrix = layer_weights[0, valid_heads].mean(dim=0).cpu().numpy()
            
            # Crop to only show rows up to and including '?' 
            cropped_matrix = avg_matrix[:q_pos + 1, :q_pos + 1]
            
            title = ' '.join(tokens)
            plot_items.append((cropped_matrix, input_tokens_str, correct_answer, title))
    
    total_plots = len(plot_items)
    if total_plots == 0:
        print("No valid sequences to visualize.")
        return
    
    n_cols = min(max_cols, total_plots)
    n_rows = math.ceil(total_plots / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols*scale, 5*n_rows*scale), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (attn_matrix, token_labels, correct_answer, title) in enumerate(plot_items):
        ax = axes_flat[idx]
        
        sns.heatmap(attn_matrix, cmap="viridis", cbar=True, square=True,
                   annot=False, ax=ax, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        ax.set_xticks(np.arange(len(token_labels)) + 0.5)
        ax.set_yticks(np.arange(len(token_labels)) + 0.5)
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(token_labels, fontsize=9)
        ax.set_xlabel('Key (attending to)', fontsize=10)
        ax.set_ylabel('Query', fontsize=10)
        
        # Find the '?' row index (last row in the cropped matrix)
        q_idx = len(token_labels) - 1
        
        # Grey out all rows before '?'
        if q_idx > 0:
            rect = mpatches.Rectangle((0, 0), len(token_labels), q_idx,
                                       linewidth=0, edgecolor='none', facecolor='white', alpha=0.6)
            ax.add_patch(rect)
        
        # Star on the highest-attended of the first two tokens
        if len(token_labels) > 1:
            vals = [attn_matrix[q_idx, 0], attn_matrix[q_idx, 1]]
            highest_c_idx = 0 if vals[0] >= vals[1] else 1
            predicted_token = token_labels[highest_c_idx]
            
            # Green if prediction matches correct answer, red if not
            star_color = "lime" if predicted_token == correct_answer else "red"
            ax.text(highest_c_idx + 0.5, q_idx + 0.5, "*", ha="center", va="center",
                    color=star_color, fontweight="bold", fontsize=20)
    
    # Hide unused subplots
    for idx in range(total_plots, n_rows * n_cols):
        axes_flat[idx].axis('off')
    
    plt.suptitle(f'Average Attention (across {len(heads)} heads)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_average_attention(model, input_list, stoi, heads, layers, config, target_length=None, save_path=None):
    """Visualize average attention patterns across multiple inputs"""
    device = next(model.parameters()).device
    
    # Check length
    if target_length:
        input_list = [inp for inp in input_list if len(inp.split()) == target_length]
        print(f"Filtered to {len(input_list)} inputs with length {target_length}")
    
    if not input_list:
        print("No valid inputs found!")
        return
    
    seq_length = len(input_list[0].split())
    attention_sums = {}
    valid_count = 0
    
    for i, input_text in enumerate(input_list):
        tokens = input_text.split()
        if len(tokens) != seq_length: continue
        
        try:
            input_tokens = [stoi[t] for t in tokens]
        except KeyError:
            continue
            
        input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            _, _, attention_weights = model(input_tensor, return_attn_weights=True)
        
        # Accumulate attention weights
        for layer_idx in layers:
            if layer_idx < len(attention_weights):
                layer_weights = attention_weights[layer_idx]
                for head_idx in heads:
                    if head_idx < layer_weights.shape[1]:
                        key = (layer_idx, head_idx)
                        attn = layer_weights[0, head_idx].cpu().numpy()
                        if key not in attention_sums:
                            attention_sums[key] = np.zeros_like(attn)
                        attention_sums[key] += attn
        
        valid_count += 1
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(input_list)} inputs")
            
    print(f"Averaged over {valid_count} valid inputs")
    
    if valid_count == 0:
        return
    
    # Compute averages and plot
    n_layers = len(layers)
    n_heads = len(heads)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(6*n_heads, 5*n_layers), squeeze=False)
    
    # Create position labels
    position_labels = [str(i+1) for i in range(seq_length)]
    
    for row, layer_idx in enumerate(layers):
        for col, head_idx in enumerate(heads):
            key = (layer_idx, head_idx)
            if key in attention_sums:
                ax = axes[row, col]
                avg_attn = attention_sums[key] / valid_count
                
                sns.heatmap(avg_attn, cmap="viridis", cbar=True, square=True,
                           annot=False, ax=ax, vmin=0, vmax=1)
                
                ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=12)
                ax.set_xticks(np.arange(seq_length) + 0.5)
                ax.set_yticks(np.arange(seq_length) + 0.5)
                ax.set_xticklabels(position_labels, fontsize=9)
                ax.set_yticklabels(position_labels, fontsize=9)
                ax.set_xlabel('Key Position', fontsize=10)
                ax.set_ylabel('Query Position', fontsize=10)
    
    plt.suptitle(f'Average Attention Patterns ({valid_count} samples, length {seq_length})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return attention_sums, valid_count


def main():
    args = parse_args()
    
    # Validate input arguments
    if not args.input and not args.inputs_file and not args.custom_inputs:
        print("Error: Must provide one of --input, --inputs_file, or --custom_inputs")
        return
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.meta_path}...")
    with open(args.meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    
    # Load model
    model, config = load_model(args.checkpoint_path, args.device)
    
    print(f"\nModel: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    print(f"Visualizing heads: {args.heads}, layers: {args.layers}")
    
    # Mode 1: Single input visualization
    if args.input:
        print(f"Single input mode: {args.input}")
        token_labels = args.input.split()
        input_tokens = []
        for token in token_labels:
            if token in stoi:
                input_tokens.append(stoi[token])
            else:
                print(f"Warning: Token '{token}' not in vocabulary, skipping")
        
        if not input_tokens:
            print("Error: No valid tokens found")
            return
        
        visualize_attention(model, input_tokens, token_labels, args.heads, args.layers, 
                           config, save_path=args.save_path)
    
    # Mode 2: Average from test file
    elif args.inputs_file:
        print(f"Inputs file mode: {args.inputs_file}")
        with open(args.inputs_file, 'r') as f:
            input_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(input_list)} inputs from file")
        
        visualize_average_attention(model, input_list, stoi, args.heads, args.layers,
                                   config, target_length=args.target_length, 
                                   save_path=args.save_path)
    
    # Mode 3: Average from custom inputs list
    elif args.custom_inputs:
        print(f"Custom inputs mode: {len(args.custom_inputs)} inputs")
        visualize_average_attention(model, args.custom_inputs, stoi, args.heads, args.layers,
                                   config, target_length=args.target_length,
                                   save_path=args.save_path)


if __name__ == '__main__':
    main()
