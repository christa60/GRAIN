import torch
from trl import SFTConfig, SFTTrainer


def min_norm_combine(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    g1_flat = g1.flatten()
    g2_flat = g2.flatten()

    diff = g2_flat - g1_flat                    # g2 - g1
    denom = torch.dot(diff, diff)               # ||g2 - g1||²

    if denom < 1e-12:
        # g1 and g2 are essentially identical — just return either
        return g1

    alpha = torch.dot(g2_flat, diff) / denom    # projection
    alpha = alpha.clamp(0.0, 1.0)              # stay on the segment

    return alpha * g1 + (1.0 - alpha) * g2


class MinNormSFTTrainer(SFTTrainer):
    """
    SFTTrainer with Min-Norm gradient combination across
    gradient_accumulation_steps=2.

    Instead of summing/averaging the two accumulated gradients,
    finds the minimum-norm convex combination (MGDA / Frank-Wolfe).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.args.gradient_accumulation_steps == 2, \
            "MinNormSFTTrainer expects gradient_accumulation_steps=2"
        self._grad_step1: dict[str, torch.Tensor] = {}
        self._accum_local_step = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ── forward + backward ──────────────────────────────────────────
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Scale by accum steps so loss magnitude stays comparable
        # (mirrors what Trainer normally does internally)
        scaled_loss = loss / self.args.gradient_accumulation_steps
        scaled_loss.backward()

        # ── gradient accumulation logic ──────────────────────────────────
        if self._accum_local_step == 0:
            # First step: snapshot gradients, then zero them so step 2
            # gets a clean slate
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self._grad_step1[name] = param.grad.clone()
            self._zero_grads(model)

        else:
            # Second step: apply min-norm combination
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                g2 = param.grad                         # current step grad
                g1 = self._grad_step1.get(name)

                if g1 is None:
                    # param only appeared in step 2 — use as-is
                    continue

                combined = min_norm_combine(g1, g2)
                param.grad.copy_(combined)

            self._grad_step1.clear()

        # Toggle which accumulation step we're on
        self._accum_local_step = 1 - self._accum_local_step

        return loss.detach()

    def _zero_grads(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def _wrap_model(self, model, training=True, dataloader=None):
        wrapped = super()._wrap_model(model, training=training, dataloader=dataloader)
        if training and hasattr(wrapped, 'register_comm_hook'):
            # Optional: also customize cross-device reduction
            wrapped.register_comm_hook(state=None, hook=self._sum_hook)
        return wrapped

    @staticmethod
    def _sum_hook(state, bucket):
        tensor = bucket.buffer()
        return (
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
            .get_future()
            .then(lambda f: f.value()[0])
        )



# ─────────────────────────────────────────────────────────────────────────────
# Global and Local version combined together
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.distributed as dist
from trl import SFTTrainer
from torch.nn.parallel import DistributedDataParallel as DDP


# ─────────────────────────────────────────────────────────────────────────────
# DDP communication hook: gather gradients from all ranks, apply min-norm
# ─────────────────────────────────────────────────────────────────────────────

def _min_norm_allgather_hook(process_group, bucket):
    """
    DDP comm hook that replaces the default all-reduce (mean) with a
    pairwise min-norm combination across 2 devices/ranks.
 
    Works for world_size == 2 only (extend reduce tree for more ranks).
 
    Flow
    ----
    1. all_gather  – every rank gets every other rank's gradient bucket
    2. pairwise min_norm_combine  – find min-norm point on the segment
    3. write result back into the bucket buffer so DDP uses it
    """
    group_to_use = process_group or dist.group.WORLD
    world_size   = dist.get_world_size(group_to_use)
    assert world_size == 2, (
        "_min_norm_allgather_hook currently supports world_size=2 only. "
        f"Got world_size={world_size}."
    )

    tensor = bucket.buffer()           # flat gradient buffer on this rank
 
    # Allocate receive buffers on the same device
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
 
    # Non-blocking all-gather, return a Future
    fut = dist.all_gather(gathered, tensor, group=group_to_use, async_op=True).get_future()
 
    def combine(fut):
        grads = fut.value()            # list of tensors from each rank
        # grads[0] = rank-0 gradients, grads[1] = rank-1 gradients
        combined = min_norm_combine(grads[0], grads[1])
        # Write in-place so DDP picks up the result
        tensor.copy_(combined)
        return [tensor]
 
    return fut.then(combine)



# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
 
class MinNormSFTTrainerGL(SFTTrainer):
    """
    SFTTrainer with **two levels** of min-norm gradient combination:
 
    Level 1 – Accumulation steps (gradient_accumulation_steps=2)
        Instead of summing the two micro-batch gradients, find the
        min-norm convex combination (MGDA / Frank-Wolfe).
 
    Level 2 – Cross-device (world_size=2, DDP)
        Instead of the default all-reduce (mean), each rank gathers the
        other rank's gradient and applies min-norm before the optimiser step.
 
    Both levels apply min_norm_combine, so the optimiser always sees the
    minimum-norm feasible gradient regardless of batch split or data split.
    """
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.args.gradient_accumulation_steps == 2, (
            "MinNormSFTTrainer expects gradient_accumulation_steps=2"
        )
        self._grad_step1: dict[str, torch.Tensor] = {}
        self._accum_local_step = 0
 
    # ── DDP hook registration ────────────────────────────────────────────────
 
    def _wrap_model(self, model, training=True, dataloader=None):
        wrapped = super()._wrap_model(model, training=training, dataloader=dataloader)
 
        if training and isinstance(wrapped, DDP):
            # Register our hook INSTEAD of the default all-reduce.
            # no_sync() is NOT used here because we want the hook to fire
            # after both accumulation steps are merged (see training_step).
            wrapped.register_comm_hook(
                state=wrapped.process_group,   # pass the process group as state
                hook=_min_norm_allgather_hook,
            )
 
        return wrapped
 
    # ── Per-step logic ───────────────────────────────────────────────────────
 
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
 
        # ── Forward + local backward ─────────────────────────────────────────
        # We manage DDP sync ourselves, so suppress automatic sync on every
        # backward pass.  The hook fires only when we exit no_sync() on the
        # SECOND accumulation step.
        ddp_model = model if isinstance(model, DDP) else None
        is_last_accum = (self._accum_local_step == 1)
 
        no_sync_ctx = (
            ddp_model.no_sync()
            if (ddp_model is not None and not is_last_accum)
            else _null_ctx()
        )
 
        with no_sync_ctx:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
 
            scaled_loss = loss / self.args.gradient_accumulation_steps
            scaled_loss.backward()
 
        # ── Level-1 min-norm: across accumulation steps ──────────────────────
        if self._accum_local_step == 0:
            # First micro-batch: snapshot gradients, zero for next step
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self._grad_step1[name] = param.grad.clone()
            self._zero_grads(model)
 
        else:
            # Second micro-batch: combine with first via min-norm
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                g1 = self._grad_step1.get(name)
                if g1 is None:
                    continue                        # only in step-2, keep as-is
                combined = min_norm_combine(g1.to(param.grad.device), param.grad)
                param.grad.copy_(combined)
 
            self._grad_step1.clear()
            # After this point the backward() above (which exited no_sync)
            # will have triggered the DDP comm hook → Level-2 min-norm fires
            # automatically across ranks.
 
        self._accum_local_step = 1 - self._accum_local_step
        return loss.detach()
 
    # ── Helpers ──────────────────────────────────────────────────────────────
 
    def _zero_grads(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Tiny null context so we don't need contextlib import at the top
# ─────────────────────────────────────────────────────────────────────────────
 
class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass