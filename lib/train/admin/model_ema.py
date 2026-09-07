"""
Exponential Moving Average (EMA) for model weights.

Usage in trainer:
    self.ema = ModelEma(self.actor.net, decay=0.9998)
    ...
    # After optimizer.step():
    self.ema.update(self.actor.net)
    ...
    # Before eval:
    self.ema.apply_shadow(self.actor.net)
    ...
    # After eval:
    self.ema.restore(self.actor.net)
"""

import copy
import torch
import torch.nn as nn


class ModelEma:
    """Model EMA (Polyak averaging) with optional warmup and device-aware storage."""

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        """
        Args:
            model: the model to track
            decay: EMA decay rate (higher = slower update, closer to 1)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        """Deep-copy all trainable parameters into shadow."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module):
        """Update shadow weights: shadow = decay*shadow + (1-decay)*model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Backup current weights and apply EMA shadow weights."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
