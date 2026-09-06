"""
Candidate elimination for computation reduction.
Only the candidate_elimination function is kept; CEABlock and other
adapter blocks are removed (single-stream does not use them).
"""
import math
import torch


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int,
                           keep_ratio: float, global_index: torch.Tensor,
                           box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction.
    Args:
        attn:         [B, num_heads, L_t + L_s, L_t + L_s]
        tokens:       [B, L_t + L_s, C]
        lens_t:       length of template
        keep_ratio:   keep ratio of search region tokens
        global_index: global index of search region tokens
        box_mask_z:   template mask for attention weight accumulation
    Returns:
        tokens_new, keep_index, removed_index
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        # Normalize to [B, N] then expand for multi-template
        if box_mask_z.dim() > 2:
            box_mask_z = box_mask_z.reshape(box_mask_z.shape[0], -1)
        if box_mask_z.shape[0] != bs:
            repeat = bs // box_mask_z.shape[0]
            box_mask_z = box_mask_z.repeat(repeat, 1)
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(
            -1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)

    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(
        dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index
