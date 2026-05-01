from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SWASFTConfig(SFTConfig):
    swa_start_epoch: float = field(
        default=0.75,
        metadata={"help": "Fraction of epochs before SWA averaging begins (0-1)."},
    )
    swa_lr: float = field(
        default=1e-5,
        metadata={"help": "Constant SWA learning rate after annealing."},
    )
    swa_anneal_epochs: int = field(
        default=1,
        metadata={"help": "Epochs to anneal LR toward swa_lr using SWALR."},
    )
    swa_anneal_strategy: str = field(
        default="cos",
        metadata={"help": "'cos' or 'linear' annealing for SWALR."},
    )
    swa_update_freq: int = field(
        default=-1,
        metadata={"help": "Steps between SWA updates. -1 means once per epoch."},
    )
    swa_update_bn_samples: int = field(
        default=0,
        metadata={"help": "Samples for BN update pass after training. 0 = skip."},
    )
    swa_save_averaged_model: bool = field(
        default=True,
        metadata={"help": "Replace final checkpoint with averaged model weights."},
    )
    swa_ema_decay: float = field(
        default=-1.0,
        metadata={"help": "EMA decay. -1 = uniform SWA. (0,1) = exponential MA."},
    )


# ---------------------------------------------------------------------------
# Callback that drives SWA logic during training
# ---------------------------------------------------------------------------

class SWACallback(TrainerCallback):

    def __init__(self, swa_config: SWASFTConfig, train_dataloader_fn: Callable):
        self.cfg = swa_config
        self._get_dataloader = train_dataloader_fn  # callable → DataLoader
        self.swa_model: Optional[AveragedModel] = None
        self.swa_scheduler: Optional[SWALR] = None
        self._swa_active = False
        self._steps_since_update = 0
        self._total_epochs: Optional[int] = None
        self._start_epoch: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _epoch_at_step(self, state: TrainerState) -> float:
        """Return fractional epoch for the current global step."""
        steps_per_epoch = max(
            state.max_steps / max(self._total_epochs, 1), 1
        )
        return state.global_step / steps_per_epoch

    def _should_update_swa(self, state: TrainerState) -> bool:
        freq = self.cfg.swa_update_freq
        if freq == -1:
            # Once per epoch: update when we crossed an epoch boundary
            return math.floor(self._epoch_at_step(state)) > math.floor(
                self._epoch_at_step(state) - 1
            )
        return (self._steps_since_update >= freq)

    @staticmethod
    def _make_avg_fn(decay: float):
        if decay > 0:
            def ema_avg_fn(averaged, current, n_averaged):
                return (decay * averaged.float() + (1.0 - decay) * current.float()).to(averaged.dtype)
            return ema_avg_fn
        else:
            def swa_avg_fn(averaged, current, n_averaged):
                a, c = averaged.float(), current.float()
                return (a + (c - a) / (n_averaged.float() + 1.0)).to(averaged.dtype)
            return swa_avg_fn

    def _init_swa(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Lazily initialise the AveragedModel and SWALR scheduler."""
        decay = self.cfg.swa_ema_decay
        avg_fn = self._make_avg_fn(decay)
        self.swa_model = AveragedModel(model, avg_fn=avg_fn)

        anneal_epochs = max(self.cfg.swa_anneal_epochs, 1)
        self.swa_scheduler = SWALR(
            optimizer,
            swa_lr=self.cfg.swa_lr,
            anneal_epochs=anneal_epochs,
            anneal_strategy=self.cfg.swa_anneal_strategy,
        )
        logger.info(
            "SWA initialised | swa_lr=%.2e | anneal_epochs=%d | strategy=%s | "
            "ema_decay=%s",
            self.cfg.swa_lr,
            anneal_epochs,
            self.cfg.swa_anneal_strategy,
            f"{decay}" if decay > 0 else "uniform",
        )

    # ------------------------------------------------------------------ #
    # TrainerCallback hooks                                                #
    # ------------------------------------------------------------------ #

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self._total_epochs = args.num_train_epochs
        self._start_epoch = self.cfg.swa_start_epoch * self._total_epochs
        logger.info(
            "SWA will activate at epoch %.2f / %.0f",
            self._start_epoch,
            self._total_epochs,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler=None,
        **kwargs,
    ):
        current_epoch = self._epoch_at_step(state)
        self._steps_since_update += 1

        # ---- Activate SWA? -----------------------------------------------
        if not self._swa_active and current_epoch >= self._start_epoch:
            self._swa_active = True
            self._init_swa(model, optimizer)
            logger.info(
                "SWA activated at epoch %.2f (step %d)",
                current_epoch,
                state.global_step,
            )

        if not self._swa_active:
            return

        # ---- Update averaged model? --------------------------------------
        if self._should_update_swa(state):
            self.swa_model.update_parameters(model)
            self._steps_since_update = 0
            logger.debug(
                "SWA model updated (step=%d, epoch=%.2f)",
                state.global_step,
                current_epoch,
            )

        # ---- Step SWALR instead of base scheduler -----------------------
        if self.swa_scheduler is not None:
            self.swa_scheduler.step()

    @staticmethod
    def _load_averaged_weights(model: torch.nn.Module, swa_module: torch.nn.Module):
        averaged_sd = swa_module.state_dict()

        # --- PEFT / LoRA path -------------------------------------------
        try:
            from peft import set_peft_model_state_dict, get_peft_model_state_dict
            # Only load the adapter parameters; ignore frozen base-layer keys.
            adapter_sd = get_peft_model_state_dict(model)
            # Keep only keys that exist in both averaged and adapter dicts.
            filtered = {k: averaged_sd[k] for k in adapter_sd if k in averaged_sd}
            missing = set(adapter_sd) - set(averaged_sd)
            if missing:
                logger.warning(
                    "SWA: %d adapter keys not found in averaged model "
                    "(they keep their current values): %s",
                    len(missing), list(missing)[:5],
                )
            result = set_peft_model_state_dict(model, filtered)
            logger.info(
                "SWA averaged adapter weights loaded via set_peft_model_state_dict "
                "(incompatible keys ignored)."
            )
            return
        except (ImportError, AttributeError):
            pass  # Not a PEFT model — fall through to plain path.

        # --- Plain model path -------------------------------------------
        incompat = model.load_state_dict(averaged_sd, strict=False)
        if incompat.missing_keys:
            logger.warning("SWA load — missing keys: %s", incompat.missing_keys[:5])
        if incompat.unexpected_keys:
            logger.warning("SWA load — unexpected keys: %s", incompat.unexpected_keys[:5])
        logger.info("SWA averaged weights loaded into model.")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        if self.swa_model is None:
            logger.warning("SWA never activated. Check swa_start_epoch setting.")
            return

        # ---- Optional BatchNorm update -----------------------------------
        if self.cfg.swa_update_bn_samples > 0:
            logger.info(
                "Updating BatchNorm statistics on SWA model (%d samples)...",
                self.cfg.swa_update_bn_samples,
            )
            loader = self._get_dataloader()
            device = next(model.parameters()).device
            self.swa_model.to(device)
            update_bn(loader, self.swa_model, device=device)
            logger.info("BatchNorm update complete.")

        # ---- Swap averaged weights into the live model -------------------
        if self.cfg.swa_save_averaged_model:
            logger.info("Copying SWA averaged weights into model for saving...")
            self._load_averaged_weights(model, self.swa_model.module)


# ---------------------------------------------------------------------------
# SWA-aware SFTTrainer
# ---------------------------------------------------------------------------

class SWASFTTrainer(SFTTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, SWASFTConfig):
            raise TypeError(
                "SWASFTTrainer requires args to be an instance of SWASFTConfig, "
                f"got {type(self.args).__name__}."
            )

        swa_callback = SWACallback(
            swa_config=self.args,
            train_dataloader_fn=self.get_train_dataloader,
        )
        self.add_callback(swa_callback)
        self._swa_callback = swa_callback
        logger.info("SWASFTTrainer initialised with SWACallback.")

    @property
    def swa_model(self) -> Optional[AveragedModel]:
        """Access the averaged model after (or during) training."""
        return self._swa_callback.swa_model

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        return super()._maybe_log_save_evaluate(*args, **kwargs)
