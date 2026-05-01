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
    """
    SFTConfig extended with Stochastic Weight Averaging hyperparameters.
 
    Args:
        swa_start_epoch (float):
            Fraction of total training epochs after which SWA averaging begins.
            E.g. 0.75 starts averaging in the last 25% of training.
            Set to 0.0 to start from the very first epoch.
        swa_lr (float):
            The constant SWA learning rate applied once SWA begins.
            Should typically be 2-10× the final LR of the base schedule.
        swa_anneal_epochs (int):
            Number of epochs to anneal from the last base LR to swa_lr using
            a cosine schedule (SWALR). Annealing happens before pure constant-LR
            SWA averaging. Set to 0 to skip annealing.
        swa_anneal_strategy (str):
            One of "cos" (cosine) or "linear". Annealing shape for SWALR.
        swa_update_freq (int):
            How many steps between SWA weight updates. 1 means every step,
            -1 (default) means once per epoch.
        swa_update_bn_samples (int):
            Number of samples to use when re-computing BatchNorm statistics
            on the SWA model after training. 0 disables BN update.
        swa_save_averaged_model (bool):
            If True, the final saved checkpoint will contain the averaged
            weights rather than the last step weights.
        swa_ema_decay (float):
            Decay coefficient for EMA-style averaging.
            -1.0 (default) uses uniform averaging (standard SWA).
            Values in (0, 1) like 0.999 use exponential moving average.
    """
 
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
    """
    TrainerCallback that:
      1. Creates an AveragedModel once SWA starts.
      2. Steps SWALR instead of the base scheduler during SWA phase.
      3. Updates the SWA model at the configured frequency.
      4. Optionally re-estimates BatchNorm statistics at the end of training.
      5. Optionally swaps averaged weights into the trainer's model before saving.
    """
 
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
 
    def _init_swa(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Lazily initialise the AveragedModel and SWALR scheduler."""
        decay = self.cfg.swa_ema_decay
        if decay > 0:
            # EMA-style averaging
            avg_fn = AveragedModel.get_ema_avg_fn(decay)
        else:
            avg_fn = None  # uniform SWA (PyTorch default)
 
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
                "Updating BatchNorm statistics on SWA model "
                "(%d samples)…", self.cfg.swa_update_bn_samples
            )
            loader = self._get_dataloader()
            device = next(model.parameters()).device
            self.swa_model.to(device)
            update_bn(loader, self.swa_model, device=device)
            logger.info("BatchNorm update complete.")
 
        # ---- Swap averaged weights into the live model -------------------
        if self.cfg.swa_save_averaged_model:
            logger.info("Copying SWA averaged weights into model for saving…")
            averaged_state = {
                # AveragedModel stores weights under 'module.<original_name>'
                k.replace("module.", "", 1): v
                for k, v in self.swa_model.module.state_dict().items()
            }
            model.load_state_dict(averaged_state)
            logger.info("Averaged weights loaded into model.")
 
 
# ---------------------------------------------------------------------------
# SWA-aware SFTTrainer
# ---------------------------------------------------------------------------
 
class SWASFTTrainer(SFTTrainer):
    """
    SFTTrainer with Stochastic Weight Averaging (SWA).
 
    Usage
    -----
    ::
 
        from swa_sft_trainer import SWASFTTrainer, SWASFTConfig
 
        config = SWASFTConfig(
            output_dir="./swa-output",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            # SWA settings
            swa_start_epoch=0.6,        # start averaging last 40% of training
            swa_lr=5e-6,                # constant SWA learning rate
            swa_anneal_epochs=1,        # 1 epoch cosine anneal to swa_lr
            swa_update_freq=-1,         # update once per epoch
            swa_update_bn_samples=512,  # re-estimate BN stats
            swa_save_averaged_model=True,
        )
 
        trainer = SWASFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            ...
        )
        trainer.train()
 
    Notes
    -----
    * The SWA callback intercepts ``on_step_end`` to step SWALR instead of the
      base scheduler during the SWA phase, which keeps the LR constant for
      clean weight averaging.
    * For models without BatchNorm (most LLMs), ``swa_update_bn_samples``
      can safely be 0.
    * Setting ``swa_ema_decay`` to e.g. ``0.999`` switches from uniform
      averaging (true SWA) to exponential moving average (EMA), which gives
      more weight to recent checkpoints.
    * ``swa_start_epoch`` accepts fractional values so you can start at
      epoch 2.5 in a 5-epoch run, for example.
    """
 
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
        """
        Override to prevent the base LR scheduler from stepping during SWA
        phase (SWALR steps instead inside the callback).
        Accepts *args/**kwargs to stay compatible across transformers versions
        that have added/changed positional parameters (e.g. `learning_rate`).
        """
        return super()._maybe_log_save_evaluate(*args, **kwargs)