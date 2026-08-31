import math

import torch


# ---------------------------------------------------------------------------
# Noam schedule — inverse-sqrt decay, learning rate tied to d_model
# ---------------------------------------------------------------------------
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    lr(t) = (1/sqrt(d_model)) * min(t^{-0.5}, t * warmup_steps^{-1.5})

    Identical to the schedule used in the AMR-Transformer paper and the original
    Attention is All You Need paper.

    Two properties make this a poor default for an architecture sweep, which is
    why ``train_transformer`` now reaches for ``WarmupCosineScheduler`` unless
    this one is asked for by name:

      * the peak learning rate scales as ``d_model ** -0.5``, so changing the
        model width silently changes the learning rate as well (9.9e-4 at
        d_model 256, 7.0e-4 at 512) and the two effects cannot be separated;
      * the inverse-sqrt tail never anneals — at 629 steps/epoch it is still
        1.8e-4 at epoch 200, having been 3.8e-4 at epoch 44.

    Kept because every run logged before 2026-08-22 used it; passing
    ``schedule="noam"`` reproduces them exactly.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 1000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * self.warmup_steps ** -1.5
        )
        return [scale for _ in self.base_lrs]


# ---------------------------------------------------------------------------
# Linear warmup into cosine decay — learning rate independent of d_model
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup to the optimizer's base LR, then cosine decay to ``eta_min``.

    Unlike ``WarmupScheduler`` the peak is whatever LR the optimizer was built
    with, so a d_model sweep varies one thing at a time. The cosine tail reaches
    ``eta_min`` at the final step, which both settles the model and makes the
    integrated learning rate — and therefore the total AdamW weight decay,
    ``exp(-weight_decay * sum(lr))`` — a stable, predictable quantity across
    runs of different length.

    Stepped once per optimizer step (not per epoch), matching the loop it is
    used in.

    Args:
        optimizer: Optimizer whose param-group ``lr`` values are the schedule's
            peak values.
        warmup_steps: Steps of linear ramp from 0 to the peak. 0 disables warmup.
        total_steps: Total optimizer steps in the run (``epochs * len(loader)``).
            The cosine reaches ``eta_min`` exactly here; later steps stay there.
        eta_min: Absolute learning-rate floor at the end of the cosine.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, eta_min: float = 1e-6):
        self.warmup_steps = max(0, warmup_steps)
        # Guard the degenerate case where the decay phase is empty, which would
        # otherwise divide by zero on the first post-warmup step.
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        if step <= self.warmup_steps:
            factor = step / self.warmup_steps
            return [base * factor for base in self.base_lrs]

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return [self.eta_min + (base - self.eta_min) * cosine for base in self.base_lrs]
