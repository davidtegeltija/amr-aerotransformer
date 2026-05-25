from typing import Any, Dict, List

import numpy as np
import torch

from src.amr.quadtree_tokenizer import QuadtreeTokenizer


class DeterministicCollateFn:
    """
    Picklable collate callable for DataLoader with num_workers > 0.
 
    For each sample in the batch, tokenizes the input grid on the CPU worker,
    then concatenates all token sequences into a single packed tensor
    (sequence packing strategy - no padding, no wasted compute).
 
    Must be a top-level class (not a closure) to be picklable by
    Python's multiprocessing.
 
    Batch dict keys:
        packed_tokens     : [total_N, C+3]             concatenated tokenized inputs
        packed_targets    : [total_N, output_channels] per-token averaged ground truth
        tokens_per_sample : List[int]                  token count per sample
    """
 
    def __init__(self, tokenizer: QuadtreeTokenizer):
        self.tokenizer = tokenizer
 
    def __call__(self, samples: List[Dict]) -> Dict:
        all_tokens = []
        all_targets = []
        tokens_per_sample = []
 
        for s in samples:
            input = s["input"]   # [H, W, C]
            target = s["target"]  # [H, W, output_channels]
 
            token_arr, leaves = self.tokenizer.tokenize(input)
 
            output_channels = target.shape[-1]
            N = len(leaves)
            token_target  = np.zeros((N, output_channels), dtype=np.float32)
            for i, node in enumerate(leaves):
                token_target[i] = target[node.r0:node.r1, node.c0:node.c1].mean(axis=(0, 1))
 
            all_tokens.append(torch.from_numpy(token_arr))
            all_targets.append(torch.from_numpy(token_target))
            tokens_per_sample.append(N)
 
        return {
            "packed_tokens":  torch.cat(all_tokens,  dim=0),
            "packed_targets": torch.cat(all_targets, dim=0),
            "tokens_per_sample":       tokens_per_sample,
        }

class LearnedCollateFn:
    """Stacks per-sample input/target grids into a batch. No tokenization."""

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        grids = torch.stack([torch.from_numpy(np.asarray(s["input"], dtype=np.float32)) for s in samples])    # [B, H, W, C]
        targets = torch.stack([torch.from_numpy(np.asarray(s["target"], dtype=np.float32)) for s in samples]) # [B, H, W, output_channels]

        return {"grids": grids, "targets": targets}
