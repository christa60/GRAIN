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