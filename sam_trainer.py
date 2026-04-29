import torch
from trl import SFTTrainer


class SAMSFTTrainer(SFTTrainer):
    def __init__(self, *args, rho=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).float()
                for p in self.model.parameters()
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ---- First forward-backward pass ----
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        # ---- Compute perturbation ----
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        e_ws = []
        for p in model.parameters():
            if p.grad is None:
                e_ws.append(None)
                continue
            # e_w = p.grad * scale
            e_w = (p.grad * scale).to(p.data.dtype)
            p.data.add_(e_w)   # ascent step
            e_ws.append(e_w)

        # ---- Second forward-backward pass ----
        self.optimizer.zero_grad()

        loss_sam = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_sam = loss_sam.mean()
        self.accelerator.backward(loss_sam)

        # ---- Restore weights ----
        for p, e_w in zip(model.parameters(), e_ws):
            if e_w is not None:
                p.data.sub_(e_w)

        return loss_sam.detach()
