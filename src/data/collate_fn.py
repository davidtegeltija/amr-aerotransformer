from typing import Dict, List

import numpy as np
import torch

from src.amr.quadtree_tokenizer import QuadtreeTokenizer


class CollateFn:
    """
    Picklable collate callable for DataLoader with num_workers > 0.
 
    For each sample in the batch, tokenizes the input grid on the CPU worker,
    then concatenates all token sequences into a single packed tensor
    (sequence packing strategy - no padding, no wasted compute).
 
    Must be a top-level class (not a closure) to be picklable by
    Python's multiprocessing.
 
    Batch dict keys:
        packed_tokens  : [total_N, C+4]            concatenated tokenized inputs
        packed_targets : [total_N, output_dim]     per-token averaged ground truth
        seq_lens       : List[int]                 token count per sample
        token_lists    : List[List[QuadNode]]      leaf nodes per sample
        grid_targets   : [B, H, W, output_dim]     full-resolution GT for evaluation
        grid_shape     : (H, W)
    """
 
    def __init__(self, tokenizer: QuadtreeTokenizer):
        self.tokenizer = tokenizer
 
    def __call__(self, samples: List[Dict]) -> Dict:
        all_tokens   = []
        all_targets  = []
        seq_lens     = []
        token_lists  = []
        grid_targets = []
 
        for s in samples:
            input = s["input"]   # [H, W, C]
            target = s["target"]  # [H, W, output_dim]
 
            token_arr, leaves = self.tokenizer.tokenize(input)
 
            H, W       = target.shape[:2]
            output_dim = target.shape[2]
            N          = len(leaves)
            token_target  = np.zeros((N, output_dim), dtype=np.float32)
            for i, node in enumerate(leaves):
                token_target[i] = target[node.r0:node.r1, node.c0:node.c1].mean(axis=(0, 1))
 
            all_tokens.append(torch.from_numpy(token_arr))
            all_targets.append(torch.from_numpy(token_target))
            seq_lens.append(N)
            token_lists.append(leaves)
            grid_targets.append(torch.from_numpy(target))
 
        return {
            "packed_tokens":  torch.cat(all_tokens,  dim=0),
            "packed_targets": torch.cat(all_targets, dim=0),
            "seq_lens":       seq_lens,
            "token_lists":    token_lists,
            "grid_targets":   torch.stack(grid_targets, dim=0),
            "grid_shape":     (H, W),
        }