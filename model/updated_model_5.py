"""
Fifth update of model.py without MLP layer + weight tying (wte - lm_head)

Key difference from updated_model_4:
  Instead of the standard "concat + W_O" multi-head attention output:
    - Each head operates in its own sub-space (dim = n_embd // n_head)
    - Head outputs are concatenated and projected through W_O (c_proj)

  This model uses "full-dim + sum":
    - Each head operates in the FULL embedding dimension (dim = n_embd)
    - Head outputs are SUMMED (averaged) across heads, no W_O projection needed
    - Q_h, K_h, V_h ∈ R^{n_embd} for each head h
    - output = (1/n_head) * Σ_h  softmax(Q_h K_h^T / √n_embd) V_h
"""
import math
import inspect
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention using the full-dim + sum approach.

    Each of the n_head heads has its own Q, K, V projection matrices of shape
    (n_embd, n_embd).  The scaled dot-product attention output from every head
    is summed (and averaged by 1/n_head) to produce the final output.
    There is NO W_O (c_proj) output projection.

    Parameter count per block:
        3 * n_head * n_embd^2  (Q, K, V for every head)
      + optional biases
    """

    def __init__(self, config):
        super().__init__()
        # Separate Q, K, V projections for each head (all full-dim)
        # We store them as a single batched linear for efficiency:
        #   shape: (n_head, n_embd, 3*n_embd) — implemented as ModuleList
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_identity_V = config.use_identity_V

        # One Linear per head for Q, K, V — each maps n_embd -> n_embd
        # We pack Q+K+V into a single (n_embd -> 3*n_embd) linear per head
        self.head_qkv = nn.ModuleList([
            nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            for _ in range(config.n_head)
        ])

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, seq_len, n_embd

        out = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)

        for h_idx, qkv_proj in enumerate(self.head_qkv):
            # Project to full-dim Q, K, V
            qkv = qkv_proj(x)                          # (B, T, 3*C)
            q, k, v = qkv.split(self.n_embd, dim=2)   # each (B, T, C)

            if self.use_identity_V:
                v = x  # identity V: use the input directly

            # Add a fake head dim so scaled_dot_product_attention can be used
            # Shape: (B, 1, T, C)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True
                )  # (B, 1, T, C)
            else:
                scale = 1.0 / math.sqrt(C)
                att = (q @ k.transpose(-2, -1)) * scale   # (B, 1, T, T)
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v                                # (B, 1, T, C)

            out = out + y.squeeze(1)  # accumulate sum over heads

        # Average over heads
        out = out / self.n_head

        out = self.resid_dropout(out)
        return out


# Removed MLP class entirely

class Block(nn.Module):
    """Transformer block without MLP - attention only"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.drop_resid = nn.Dropout(config.dropout)

    def forward(self, x):
        shortcut = x
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_identity_embeddings: bool = False
    use_fixed_positions: bool = False
    use_identity_V: bool = False  # Whether to use identity matrix for V projection

class IdentityEmbedding(nn.Module):
    """Identity embedding layer - creates one-hot vectors for input tokens, extended to n_embd dimensions"""
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        if n_embd == vocab_size:
            identity_weight = torch.eye(vocab_size)
        elif n_embd > vocab_size:
            identity_weight = torch.zeros(vocab_size, n_embd)
            identity_weight[:, :vocab_size] = torch.eye(vocab_size)
        else:
            identity_weight = torch.zeros(vocab_size, n_embd)
            identity_weight[:n_embd, :] = torch.eye(n_embd)

        self.register_buffer('weight', identity_weight)

    def forward(self, idx):
        return F.embedding(idx, self.weight)

class FixedPositionalEmbedding(nn.Module):
    """Fixed positional embedding using identity matrix (one-hot encoding)"""
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        identity_matrix = torch.eye(block_size)
        self.register_buffer('position_embeddings', identity_matrix)

    def forward(self, seq_len):
        return self.position_embeddings[:seq_len]

class IdentityLinear(nn.Module):
    """Identity linear layer that uses the transposed extended embedding matrix"""
    def __init__(self, embedding_weight, bias=False):
        super().__init__()
        self.register_buffer('weight', embedding_weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(embedding_weight.size(0)))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Calculate effective embedding dimension
        if config.use_fixed_positions:
            token_emb_dim = config.n_embd - config.block_size
            if token_emb_dim <= 0:
                raise ValueError(
                    f"n_embd ({config.n_embd}) must be larger than block_size ({config.block_size}) "
                    "when using fixed positions"
                )
        else:
            token_emb_dim = config.n_embd

        if config.use_identity_embeddings:
            embedding_dim = config.n_embd
        else:
            embedding_dim = token_emb_dim

        self.transformer = nn.ModuleDict(dict(
            wte=(nn.Embedding(config.vocab_size, embedding_dim)
                 if not config.use_identity_embeddings
                 else IdentityEmbedding(config.vocab_size, embedding_dim)),
            wpe=(FixedPositionalEmbedding(config.block_size)
                 if config.use_fixed_positions
                 else nn.Embedding(config.block_size, config.n_embd)),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        if config.use_identity_embeddings:
            self.lm_head = IdentityLinear(self.transformer.wte.weight, bias=False)
            print("Weight tying: On (Identity)")
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            if not config.use_fixed_positions:
                self.transformer.wte.weight = self.lm_head.weight
                print("Weight tying: On (Learned)")
            else:
                print("Weight tying: Off (dimension mismatch)")

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        self.report_parameter_stats()

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_fixed_positions:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def report_parameter_stats(self):
        total_params = sum(p.numel() for p in self.parameters())

        embeddings = 0
        attention = 0
        layernorm = 0
        head = 0
        others = 0

        for pn, p in self.named_parameters():
            n = p.numel()
            if 'wte' in pn or 'wpe' in pn:
                embeddings += n
            elif 'attn' in pn:
                attention += n
            elif 'ln_' in pn:
                layernorm += n
            elif 'lm_head' in pn:
                head += n
            else:
                others += n

        def format_params(num):
            if num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return f"{num}"

        breakdown_str = f"Emb: {format_params(embeddings)} | Attn: {format_params(attention)} | LN: {format_params(layernorm)}"
        if head > 0: breakdown_str += f" | Head: {format_params(head)}"
        if others > 0: breakdown_str += f" | Others: {format_params(others)}"

        print(f"Number of parameters: {format_params(total_params)} ({breakdown_str})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

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

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

        if self.config.use_fixed_positions:
            new_identity = torch.eye(block_size)
            self.transformer.wpe.register_buffer('position_embeddings', new_identity)
        else:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding, IdentityEmbedding, IdentityLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        if 'lm_head.weight' in decay:
            decay.remove('lm_head.weight')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 bfloat16 peak TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
