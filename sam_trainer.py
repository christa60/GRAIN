import torch
from transformers import Trainer


class SAMTrainer(Trainer):
    def __init__(self, *args, rho=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for p in self.model.parameters()
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ---- First forward-backward pass ----
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # loss.backward()
        self.accelerator.backward(loss)

        # ---- Compute perturbation ----
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        e_ws = []
        for p in model.parameters():
            if p.grad is None:
                e_ws.append(None)
                continue
            e_w = p.grad * scale
            p.data.add_(e_w)   # ascent step
            e_ws.append(e_w)

        # ---- Second forward-backward pass ----
        self.optimizer.zero_grad()

        loss_sam = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_sam = loss_sam.mean()
        # loss_sam.backward()
        self.accelerator.backward(loss_sam)

        # ---- Restore weights ----
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is not None:
                p.data.sub_(e_w)

        return loss_sam.detach()