"""
lr_scheduler.py — Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017), §5.3

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

The autograder checks:
  • LR is monotonically increasing during warm-up
  • Peak occurs within 10 steps of warmup_steps
  • LR is monotonically decreasing after warm-up
  • Peak value matches the closed-form formula
  • LR at step 1 matches the formula

Note: For the formula to produce the actual returned LR, every param group's
`lr` must be set to 1.0 BEFORE constructing this scheduler — i.e. Adam(...
lr=1.0). Then base_lr = 1.0 and the scheduler's scale becomes the LR.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    Noam scheduler. Multiplies each param group's `base_lr` by

        d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Construct AFTER setting Adam(lr=1.0).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.d_model      = int(d_model)
        self.warmup_steps = int(warmup_steps)
        # NOTE: parent __init__ immediately calls get_lr() with last_epoch=0,
        # so attributes above must be set first.
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        # last_epoch starts at 0 after __init__; first .step() makes it 1.
        # Spec: "step 1" must match formula, so we use max(self.last_epoch, 1).
        step = max(self.last_epoch, 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def get_lr(self) -> list:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
# Helper — do NOT modify
# ──────────────────────────────────────────────────────────────────────

def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list:
    """Simulate the LR trajectory for `total_steps` steps."""
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()