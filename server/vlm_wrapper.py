import re
import torch
import numpy as np

_READING_CTX_PATTERN = re.compile(
    r'<<<READING_CONTEXT_START>>>\n.*?\n<<<READING_CONTEXT_END>>>\n',
    re.DOTALL,
)
from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
from streaming_vlm.inference.streaming_args import StreamingArgs
from streaming_vlm.inference.inference import process_past_kv
from streaming_vlm.utils.get_qwen_range import TOKEN_IDS

class HanLabStreamingVLM:
    """
    Wrapper for StreamingVLM logic based on streaming_vlm/inference/inference.py.
    Manages KV cache windowing and iterative video-text generation.
    """
    def __init__(self, model, processor, device):
        # Patch the model for streaming attention
        self.model = convert_qwen2_5_to_streaming(model)

        # Manual attribute injection to fix compatibility with newer transformers versions
        for m in self.model.modules():
            if "Attention" in m.__class__.__name__ and not hasattr(m, "_flash_attn_uses_top_left_mask"):
                m._flash_attn_uses_top_left_mask = False

        self.processor = processor
        self.device = device
        
        # State management variables
        self.past_key_values = None
        self.full_conversation_history = []
        self.prev_generated_ids = None
        self.recent_video_window_clips = []
        self.recent_pixel_values_videos = []
        self.chunk_index = 0
        
        # Streaming configuration
        self.streaming_args = StreamingArgs(pos_mode="shrink", all_text=False)
        self.assistant_start_bias = len(processor(text="<|im_start|>assistant\n")['input_ids'][0])
        self.assistant_end_bias = len(processor(text=" ...<|im_end|>")['input_ids'][0])
        # System prompt offset for chat template
        self.system_prompt_offset = len(processor.apply_chat_template([{"role": "system", "content": ""}], tokenize=False))
        self.frame_buffer = []
        # Match inference.py defaults
        self.text_round = 16
        self.visual_round = 16
        self.max_new_tokens = 20
        self.base_instruction = "Short noun phrases only, comma-separated. No full sentences. Example: chair left, bottle on table, socks floor, person walking."

    def strip_reading_context(self):
        """Remove injected reading-session OCR text from conversation history and invalidate KV cache."""
        modified = False
        for turn in self.full_conversation_history:
            if turn.get("role") != "user":
                continue
            content = turn["content"]
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text" and "<<<READING_CONTEXT_START>>>" in item["text"]:
                        item["text"] = _READING_CTX_PATTERN.sub("", item["text"])
                        modified = True
            elif isinstance(content, str) and "<<<READING_CONTEXT_START>>>" in content:
                turn["content"] = _READING_CTX_PATTERN.sub("", content)
                modified = True
        if modified:
            self.past_key_values = None
            self.prev_generated_ids = None
            print("[VLM] Stripped reading context from history. KV cache invalidated.")

    def reset(self):
        self.past_key_values = None
        self.full_conversation_history = []
        self.prev_generated_ids = None
        self.recent_video_window_clips = []
        self.recent_pixel_values_videos = []
        self.chunk_index = 0
        self.frame_buffer = []
        self.streaming_args.input_ids = None
        self.streaming_args.video_grid_thw = None
        self.streaming_args.second_per_grid_ts = None
        print("[VLM] KV cache and conversation history cleared.")

    def update_params(self, params: dict):
        self.max_new_tokens = params.get("max_new_tokens", self.max_new_tokens)
        self.base_instruction = params.get("base_instruction", self.base_instruction)
        self.text_round = params.get("text_round", self.text_round)
        self.visual_round = params.get("visual_round", self.visual_round)

    def push_frame(self, pil_image):
        """Buffers frames for the next streaming inference step."""
        self.frame_buffer.append(np.array(pil_image))

    def chat(self, query):
        """Interface for user queries. Triggers a video-text inference step immediately."""
        return self.process_video_step(query)

    def process_video_step(self, query=None):
        """
        Performs a single streaming inference step.
        If query is None, generates a background description to serve as memory.
        """
        # Guard: skip entirely if no frames — don't touch KV state so the next
        # call with the same chunk_index can still prune + refill correctly.
        if len(self.frame_buffer) == 0:
            return ""

        # 1. Manage KV Cache and sliding window
        self.past_key_values, self.prev_generated_ids, self.recent_video_window_clips, self.recent_pixel_values_videos = process_past_kv(
            self.past_key_values, self.chunk_index,
            text_round=self.text_round, visual_round=self.visual_round,
            full_conversation_history=self.full_conversation_history,
            prev_generated_ids=self.prev_generated_ids,
            assistant_start_bias=self.assistant_start_bias,
            assistant_end_bias=self.assistant_end_bias,
            recent_video_window_clips=self.recent_video_window_clips,
            recent_pixel_values_videos=self.recent_pixel_values_videos,
            text_sink=512, text_sliding_window=512
        )

        # 2. Pull the 1-second chunk of frames from the buffer
        # Take all frames buffered in the last second
        current_clip = self.frame_buffer
        self.frame_buffer = [] # Reset buffer for next step
        self.recent_video_window_clips.append(current_clip)
        
        # 3. Build text prompt for the current step
        prompt = f'Time={self.chunk_index:.1f}-{self.chunk_index+1.0:.1f}s'
        
        # Base instruction acts as the constant 'attention sink' task description
        base_instruction = self.base_instruction

        if self.chunk_index == 0:
            # Always include the base instruction in the first chunk to anchor the task
            user_content = [{"type": "text", "text": prompt}, {"type": "video", "video": "live"}, {"type": "text", "text": base_instruction}]
            if query:
                user_content.append({"type": "text", "text": query})
                
            self.full_conversation_history = [{"role": "previous text", "content": ""}, {"role": "user", "content": user_content}]
            text = self.processor.apply_chat_template(self.full_conversation_history, tokenize=False, add_generation_prompt=True)
        else:
            user_content = [{"type": "text", "text": prompt}, {"type": "video", "video": "live"}]
            if query:
                user_content.append({"type": "text", "text": query})
            self.full_conversation_history.append({"role": "user", "content": user_content})
            text = self.processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            text = '\n' + text[self.system_prompt_offset:]

        # 4. Prepare model inputs
        inputs = self.processor(text=[text], videos=self.recent_video_window_clips[-1], padding=True, return_tensors="pt").to(self.device)
        if self.prev_generated_ids is not None:
            if self.prev_generated_ids[:,-1].item() != TOKEN_IDS["\n"]:
                inputs['input_ids'] = torch.cat([self.prev_generated_ids, inputs['input_ids']], dim=1)
            else:
                inputs['input_ids'] = torch.cat([self.prev_generated_ids, inputs['input_ids'][:, 1:]], dim=1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        self.recent_pixel_values_videos.append(inputs['pixel_values_videos'])
        
        # Sync streaming args for position embedding recomputation (shrink mode)
        self.streaming_args.input_ids = inputs['input_ids']
        self.streaming_args.video_grid_thw = inputs['video_grid_thw'] if self.chunk_index == 0 else \
            torch.cat([self.streaming_args.video_grid_thw, inputs['video_grid_thw']], dim=0)

        # 5. Generate Response
        outputs = self.model.generate(**inputs, past_key_values=self.past_key_values, max_new_tokens=self.max_new_tokens, 
                                      use_cache=True, return_dict_in_generate=True, do_sample=True,
                                      streaming_args=self.streaming_args, pad_token_id=151645,
                                      temperature=0.9, repetition_penalty=1.05)
        
        generated_ids = outputs.sequences
        # Mimic inference.py: Ensure sequence ends with <|im_end|> for consistent KV structure
        # Mimic inference.py: Ensure sequence ends with <|im_end|> (151645) for consistent KV structure
        if generated_ids[0, -1].item() != 151645:
            generated_ids = torch.cat([generated_ids, torch.tensor([[151645]], device=self.device)], dim=1)

        newly_generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(newly_generated_ids, skip_special_tokens=True)[0]

        # Mimic inference.py memory logic: ensure response ends with " ..." 
        # This allows process_past_kv to correctly identify and prune text for the memory sink.
        if not response.endswith(" ..."):
            response += " ..."

        # 6. Update state for next step
        self.past_key_values = outputs.past_key_values
        self.prev_generated_ids = generated_ids.clone()
        self.full_conversation_history.append({"role": "assistant", "content": response})
        self.chunk_index += 1
        return response