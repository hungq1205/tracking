from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
from types import MethodType
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger,apply_rotary_pos_emb_flashatt,flash_attn_varlen_func

def streaming_visual_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

def streaming_visual_block_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

    return hidden_states


def streaming_visual_encoder_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Vision backbone: patch-embed → window reordering → multi-layer local/global attention → merge"""

    # 1. Patch → Token
    hidden_states = self.patch_embed(hidden_states)

    # 2. Rotary position encoding
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    # 3. window indexing
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)  # Remove duplicate window start points

    # 4. Rearrange tokens within the same window to adjacent memory blocks
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit,
                                          self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index]
    hidden_states = hidden_states.reshape(seq_len, -1)

    # Position encoding is also reordered
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit,
                                            self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index].reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # 5. Global cu_seqlens
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                         grid_thw[:, 0]).cumsum(dim=0).to(torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0))

    # 6. Swin blocks
    for idx, blk in enumerate(self.blocks):
        cu_now = cu_seqlens if idx in self.fullatt_block_indexes else cu_window_seqlens
        hidden_states = blk(hidden_states, cu_seqlens=cu_now, position_embeddings=position_embeddings)

    # 7. Merge + restore token order
    hidden_states = self.merger(hidden_states)
    hidden_states = hidden_states[torch.argsort(window_index)]

    return hidden_states

