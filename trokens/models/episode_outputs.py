"""Small shared output utilities, independent of historical routing encoders."""
import torch


def _merge_outputs(parts, class_axis=False):
    if not parts or not parts[0]:
        return {}
    result = {}
    for key in parts[0]:
        values = [part[key] for part in parts]
        if key.endswith('reference_available'):
            result[key] = torch.cat(values, 0) if class_axis else values[0]
        else:
            result[key] = torch.cat(values, 1 if class_axis else 0)
    return result


def _detach_diagnostics(result):
    """Only losses are DDP autograd roots; do not re-enter CUDA checkpoints
    concurrently through unused diagnostic outputs (PyTorch 2.0 DDP behavior).
    Detach at the final boundary, after all differentiable logits are computed.
    """
    loss_inputs = {
        'query_partial_q2s_logits', 'query_partial_q2s_base_logits',
        'query_quality_logits', 'query_relation_logits',
        'global_classification_logits',
    }
    return {key: value.detach() if torch.is_tensor(value) and key not in loss_inputs else value
            for key, value in result.items()}
