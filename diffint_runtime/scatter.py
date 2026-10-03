"""Native PyTorch reductions for DiffInt's dim=0, 1-D batch indices."""
import torch


def scatter_add(src, index, dim=0, dim_size=None):
    if dim != 0 or index.ndim != 1 or index.numel() != src.shape[0]:
        raise ValueError("Expected dim=0 and one batch index per source row")
    size = int(index.max()) + 1 if index.numel() else 0
    if dim_size is not None:
        size = dim_size
    shape = (size, *src.shape[1:])
    expanded = index.reshape((-1,) + (1,) * (src.ndim - 1)).expand_as(src)
    return src.new_zeros(shape).scatter_add_(0, expanded, src)


def scatter_mean(src, index, dim=0, dim_size=None):
    result = scatter_add(src, index, dim=dim, dim_size=dim_size)
    counts = scatter_add(src.new_ones(index.shape), index, dim_size=result.shape[0])
    counts = counts.clamp_min(1).reshape((-1,) + (1,) * (src.ndim - 1))
    return result / counts if result.is_floating_point() else result.div(counts, rounding_mode="floor")
