"""Optional scalar-libm compatibility for the validated historical CPU build.
The first-order derivatives follow PyTorch 2.0.1's activation expressions.
No old PyTorch library is loaded. Higher-order gradients are unsupported.
"""
import numpy as np
import torch
from torch.autograd.function import once_differentiable


def _libm(x, operation):
    if x.device.type != 'cpu' or x.dtype != torch.float32:
        raise ValueError('Old-compatible math requires CPU float32; use --no-old-compatible for standard math')
    from . import _legacy_math
    source = np.ascontiguousarray(x.detach().numpy())
    output = np.empty_like(source)
    getattr(_legacy_math, operation)(source, output)
    return torch.from_numpy(output).reshape(x.shape)


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind):
        ctx.kind = kind
        if kind == 'tanh':
            output = _libm(x, 'tanh')
            ctx.save_for_backward(output)
            return output
        exponential = _libm(-x, 'exp')
        sigmoid = 1 / (1 + exponential)
        ctx.save_for_backward(x, sigmoid)
        return x / (1 + exponential) if kind == 'silu' else sigmoid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if ctx.kind == 'tanh':
            output, = ctx.saved_tensors
            return grad * (1 - output * output), None
        x, sigmoid = ctx.saved_tensors
        if ctx.kind == 'silu':
            return grad * sigmoid * (1 + x * (1 - sigmoid)), None
        return grad * (1 - sigmoid) * sigmoid, None


class CompatibleSiLU(torch.nn.SiLU):
    def forward(self, x):
        return _Activation.apply(x, 'silu')


class CompatibleSigmoid(torch.nn.Sigmoid):
    def forward(self, x):
        return _Activation.apply(x, 'sigmoid')


def tanh(x, old_compatible):
    return _Activation.apply(x, 'tanh') if old_compatible else torch.tanh(x)


def cross(x, y, old_compatible):
    if not old_compatible:
        return torch.cross(x, y, dim=1)
    return torch.stack([x[:, 1]*y[:, 2] - x[:, 2]*y[:, 1],
                        x[:, 2]*y[:, 0] - x[:, 0]*y[:, 2],
                        x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]], dim=1)


def configure_math(module, old_compatible):
    for parent in list(module.modules()):
        parent.old_compatible = old_compatible
        for name, child in list(parent._modules.items()):
            if isinstance(child, torch.nn.SiLU):
                parent._modules[name] = CompatibleSiLU() if old_compatible else torch.nn.SiLU()
            elif isinstance(child, torch.nn.Sigmoid):
                parent._modules[name] = CompatibleSigmoid() if old_compatible else torch.nn.Sigmoid()
