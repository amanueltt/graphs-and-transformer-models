import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.updated_model_5 import GPTConfig, GPT  # Updated import for full-dim + sum model
import numpy as np
import argparse
import pickle
import re
import torch
from tqdm import tqdm
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_iter', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='list', help='Should be "list" for list reversal task')
    parser.add_argument('--config', type=str, default='1_1_120')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--min_value', type=int, default=0)
    parser.add_argument('--max_value', type=int, default=100)
    parser.add_argument('--is_sorted', type=str, default="True", help='Whether lists are sorted')
    parser.add_argument('--num_list_copies', type=int, default=5)
    parser.add_argument('--embedding_config', type=str, default='', help='Optional suffix for embedding config')
    parser.add_argument('--test_samples', type=int, default=-1, help='Number of test samples to evaluate (-1 for all)')
    parser.add_argument('--fixed_length', type=int, default=None, help='Fixed length of lists if specified')
    parser.add_argument('--permutation_type', type=str, default="reversal", 
                        choices=["reversal", "random", "manual", "custom", "copy"],
                        help='Type of permutation to apply (reversal, random, manual, custom, copy)')
    parser.add_argument('--no_separator', action='store_true', 
                        help='Test data has no % separator, expected output is last token')
    return parser.parse_args()

def encode(s, stoi):
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string

def decode(l, itos):
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]

def check_prediction(generated, expected_output, no_separator=False):
    """Simple check if prediction matches expected output"""
    
    if no_separator:
        generated_tokens = generated.strip().split()
        if not generated_tokens:
            return False
        last_token = generated_tokens[-1]
        return last_token == expected_output
    
    # For separator case - extract prediction after %
    if '%' not in generated:
        return False
    
    parts = generated.split('%')
    if len(parts) < 2:
        return False
    
    model_output = parts[-1].strip()  # Take last part after final %
    return model_output == expected_output

def main():
    args = parse_args()
    
    # Determine list type directory and path structure
    list_type = "sorted" if args.is_sorted == "True" else "unsorted"
    embedding_config = args.embedding_config
    full_config = f'{args.config}_{embedding_config}' if embedding_config else args.config
    
    # Define paths with new directory structure
    length_type = f"fixed{args.fixed_length}" if args.fixed_length is not None else "variable"
    data_path = f'data/{args.dataset}/{list_type}/{length_type}/{args.min_value}-{args.max_value}/{args.permutation_type}'
    meta_path = f'{data_path}/meta.pkl'
    
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    max_new_tokens = meta['block_size']
    top_k = len(stoi)
    
    # Updated output directory structure
    out_dir = f'out/{args.dataset}_{list_type}_{length_type}_{args.permutation_type}_{full_config}_{args.min_value}-{args.max_value}'
    os.makedirs(out_dir, exist_ok=True)
    
    if args.num_list_copies == 0:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'{args.ckpt_iter}_ckpt_{args.num_list_copies}.pt')
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    # model_5 GPTConfig does not have use_identity_output_projection — drop it if present
    model_args = checkpoint['model_args'].copy()
    model_args.pop('use_identity_output_projection', None)

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    # Print model configuration
    print(f"Model configuration:")
    print(f"- Layers: {gptconf.n_layer}")
    print(f"- Heads: {gptconf.n_head}")
    print(f"- Embedding dim: {gptconf.n_embd}")
    print(f"- Using identity embeddings: {getattr(gptconf, 'use_identity_embeddings', False)}")
    print(f"- Using fixed positions: {getattr(gptconf, 'use_fixed_positions', False)}")
    print(f"- Using identity V: {getattr(gptconf, 'use_identity_V', False)}")
    print(f"- Attention: full-dim + sum (no W_O projection)")
    print(f"- Testing with permutation type: {args.permutation_type}")
    print(f"- No separator mode: {args.no_separator}")
    
    model.eval()
    model.to(args.device)
    
    # Load test data
    test_file = f'{data_path}/test.txt'
    print(f"Loading test data from {test_file}...")
    
    test_prompts = []
    test_expected_outputs = []
    
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Always use all tokens except last as prompt, last token as expected
            tokens = line.split()
            if len(tokens) >= 2:
                # Use all tokens except the last as prompt
                prompt_tokens = tokens[:-1]
                expected_token = tokens[-1]  # Only test the last token
                
                test_prompts.append(" ".join(prompt_tokens))
                test_expected_outputs.append(expected_token)
    
    print(f"Loaded {len(test_prompts)} test examples")
    
    # Limit test samples if specified
    if args.test_samples > 0 and args.test_samples < len(test_prompts):
        import random
        random.seed(42)
        indices = random.sample(range(len(test_prompts)), args.test_samples)
        test_prompts = [test_prompts[i] for i in indices]
        test_expected_outputs = [test_expected_outputs[i] for i in indices]
        print(f"Limited testing to {args.test_samples} randomly sampled examples")
    
    # Convert prompts to tensors individually
    encoded_prompts = [encode(p, stoi) for p in test_prompts]
    encoded_texts = [torch.tensor(p, dtype=torch.long, device=args.device) for p in encoded_prompts]
    
    # Create output file
    output_file = os.path.join(out_dir, f'list_{args.permutation_type}_{args.ckpt_iter}.txt')
    with open(output_file, 'w') as f:
        pass
    
    total = 0
    correct = 0
    unique_sequences = set()
    unique_correct = set()
    
    batch_size = min(100, len(encoded_texts))
    total_samples = len(encoded_texts)
    
    print(f"Testing on {total_samples} examples")
    if args.no_separator:
        print("Note: Testing last token prediction (no separator mode)")
    else:
        print("Note: Testing with % separator")
    
    for i in tqdm(range(0, total_samples, batch_size)):
        end_idx = min(i + batch_size, total_samples)
        
        y_pred = []
        for j in range(i, end_idx):
            x = encoded_texts[j].unsqueeze(0)
            y = model.generate(x, 1, temperature=args.temperature, top_k=top_k)
            y_pred.append(decode(y[0].tolist(), itos).split('\n')[0])
        
        with open(output_file, 'a') as f:
            for j, item in enumerate(y_pred):
                prompt_idx = i + j
                prompt = test_prompts[prompt_idx]
                expected_output = test_expected_outputs[prompt_idx]
                
                unique_sequences.add(prompt)
                
                generated_tokens = item.strip().split()
                predicted_token = generated_tokens[-1] if generated_tokens else "no_output"
                
                is_correct = (predicted_token == expected_output)
                total += 1
                if is_correct:
                    correct += 1
                    unique_correct.add(prompt)
                
                if is_correct:
                    f.write(f"{prompt} -> {predicted_token}\n")
                else:
                    f.write(f"{prompt} -> {predicted_token} (expected: {expected_output}) [x]\n")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\nTest Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Unique sequences correct: {len(unique_correct)}/{len(unique_sequences)}")
    print(f"Temperature: {args.temperature}")
    
    with open(output_file, 'a') as f:
        f.write("\n" + "-"*50 + "\n")
        f.write("SUMMARY:\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total: {total}, Correct: {correct}, Wrong: {total - correct}\n")
        f.write(f"Unique sequences correct: {len(unique_correct)}/{len(unique_sequences)}\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
