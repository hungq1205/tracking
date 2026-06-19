
from typing import Optional, Tuple
import torch
from qwen_vl_utils.vision_process import FPS

def get_rope_index(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1
        Changed to second_per_grid_t = 2 / FPS

        Calculate 3D RoPE indices:
            - Generate (t, h, w) 3D position indices for vision patches
            - Generate 1D position indices for text tokens (replicated across three dimensions)
        """
        spatial_merge_size      = self.config.vision_config.spatial_merge_size
        image_token_id          = self.config.image_token_id
        video_token_id          = self.config.video_token_id
        vision_start_token_id   = self.config.vision_start_token_id
        mrope_position_deltas   = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids

            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)

            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float32,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]

                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens        = input_ids[vision_start_indices + 1]
                image_nums           = (vision_tokens == image_token_id).sum()
                video_nums           = (vision_tokens == video_token_id).sum()

                input_tokens   = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    ed_image = input_tokens.index(image_token_id, st)  if image_token_id in input_tokens and remain_images > 0 else len(input_tokens) + 1
                    ed_video = input_tokens.index(video_token_id, st)  if video_token_id in input_tokens and remain_videos > 0 else len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0
                        image_index  += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        second_per_grid_t = 2 / FPS
                        video_index  += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t.item()
                    llm_grid_h = h.item() // spatial_merge_size
                    llm_grid_w = w.item() // spatial_merge_size
                    text_len   = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor    = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range  = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    second_per_grid_t = torch.as_tensor(second_per_grid_t, dtype=torch.float32, device=range_tensor.device)
                    time_tensor     = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    t_index         = time_tensor.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )

                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device).to(torch.float32)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas

        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)

                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas
