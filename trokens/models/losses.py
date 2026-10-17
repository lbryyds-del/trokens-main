"""Classification losses used by the maintained dataset entrypoints."""

from torch import nn


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
}


def get_loss_func(cfg, state="train"):
    loss_name = cfg.MODEL.LOSS_FUNC
    if loss_name not in _LOSSES:
        raise NotImplementedError(f"Loss {loss_name} is not supported")
    return _LOSSES[loss_name]
