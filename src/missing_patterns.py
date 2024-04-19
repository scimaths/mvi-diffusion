import torch

def block_mcar(X, mask_percent=0.1, block_size=10):
    X = torch.clone(X)
    num_series, series_len = X.shape

    total_blocks = series_len // (2 * block_size)
    block_mask_shape = (num_series, total_blocks)
    block_mask = torch.rand(block_mask_shape) < 2 * mask_percent

    offsets = torch.randint(low=0, high=block_size + 1, size=block_mask_shape).unsqueeze(-1)
    offset_mask = torch.arange(2 * block_size).reshape(1, 1, -1).repeat(num_series, total_blocks, 1)
    offset_mask = (offset_mask >= offsets) * (offset_mask < offsets + block_size)

    residue_mask = torch.zeros(num_series, series_len % (2 * block_size))
    complete_block_mask = torch.cat([
        (block_mask.unsqueeze(-1) * offset_mask).reshape(num_series, -1),
        residue_mask
    ], dim=-1).bool()
    X[complete_block_mask] = torch.nan
    return X

def get_disjoint_mask(num_series, series_len):
    block_size = series_len // num_series
    block_indices = torch.arange(series_len).reshape(1, -1).repeat(num_series, 1) // block_size
    block_mask = (block_indices % num_series) == torch.arange(num_series).unsqueeze(-1).repeat(1, series_len)
    return block_mask

def block_missing_disjoint(X):
    X = torch.clone(X)
    num_series, series_len = X.shape
    block_mask = get_disjoint_mask(num_series, series_len)
    X[block_mask] = torch.nan
    return X

def block_missing_overlap(X):
    X = torch.clone(X)
    num_series, series_len = X.shape
    block_mask = get_disjoint_mask(num_series, series_len)
    block_mask[:-1, :] = block_mask[:-1, :] + block_mask[1:, :]
    X[block_mask] = torch.nan
    return X

def blackout(X, mask_start=0.3, mask_percent=0.1):
    assert mask_start + mask_percent < 1
    X = torch.clone(X)
    num_series, series_len = X.shape
    mask_start_index = int(series_len * mask_start)
    mask_end_index = int(series_len * (mask_start + mask_percent))
    series_arange = torch.arange(series_len).unsqueeze(0).repeat(num_series, 1)
    block_mask = (series_arange >= mask_start_index) * (series_arange < mask_end_index)
    X[block_mask] = torch.nan
    return X