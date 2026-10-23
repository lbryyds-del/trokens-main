"""Process-local cache of frozen text-tower outputs, BEFORE trainable projection."""

import torch


class FrozenLabelTextCache:
    """Cache prompt groups without adding checkpoint state or detaching a projection.

    Each rank fills its own cache lazily. Parameter/buffer versions, device,
    precision and tokenizer identity invalidate it; prompt strings are the keys.
    No assumptions are made about episode class order or train/test label splits.
    """

    def __init__(self):
        self.context = None
        self.features = {}

    def clear(self):
        self.context = None
        self.features.clear()

    @staticmethod
    def _encode(encoder, tokenizer, prompts, device):
        # A cache first filled in inference_mode must remain usable later in
        # training, including by a trainable text_to_model_proj.
        with torch.inference_mode(False), torch.no_grad():
            tokens = tokenizer.tokenize(prompts).to(device, non_blocking=True)
            features = encoder(tokens).float()
            return features[:, features.shape[-1] // 2:].clone()

    def encode(self, encoder, tokenizer, prompt_groups):
        parameters = tuple(encoder.parameters())
        device = parameters[0].device
        encoder.eval()
        groups = [tuple(group) for group in prompt_groups]
        if any(p.requires_grad for p in parameters):
            # Never persist outputs of a changing tower. The existing caller's
            # no-grad tower contract is preserved even when cache is bypassed.
            self.clear()
            return self._encode(encoder, tokenizer, [p for g in groups for p in g], device)
        autocast = ((torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())
                    if device.type == 'cuda' else
                    (torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()))
        tensors = parameters + tuple(encoder.buffers())
        context = (
            id(encoder), id(tokenizer), autocast,
            torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision(),
            tuple((id(t), t._version, t.device, t.dtype, tuple(t.shape)) for t in tensors),
        )
        if context != self.context:
            self.clear()
            self.context = context
        missing = list(dict.fromkeys(group for group in groups if group not in self.features))
        if missing:
            encoded = self._encode(encoder, tokenizer, [p for g in missing for p in g], device)
            for group, value in zip(missing, encoded.split([len(g) for g in missing])):
                self.features[group] = value
        # Also keep the concatenated result a normal tensor when the first
        # request happens under inference_mode. No cached value has a grad_fn.
        with torch.inference_mode(False):
            return torch.cat([self.features[group] for group in groups], dim=0)
