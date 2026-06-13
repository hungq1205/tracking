from streaming_vlm.inference.qwen2.vision_forward import streaming_visual_attention_forward,streaming_visual_block_forward, streaming_visual_encoder_forward
from streaming_vlm.inference.qwen2.language_forward import streaming_language_model_forward,streaming_text_attn_forward,streaming_text_decoder_layer_forward
from streaming_vlm.inference.qwen2.model_forward import model_forward,qwen2_vl_forward,prepare_inputs_for_streaming_generation
from streaming_vlm.inference.qwen2.pos_emb import get_rope_index
from transformers import Qwen2VLForConditionalGeneration,Qwen2VLProcessor,GenerationMixin
from types import MethodType
from collections import deque
from streaming_vlm.inference.streaming_args import StreamingArgs
from streaming_vlm.inference.generate.streaming_generate_qwen import streaming_generate,_sample
from streaming_vlm.inference.generate.prepare_generation import prepare_multiturn_multimodal_inputs_for_generation
import ffmpeg
import torch
import os, torchvision
import re
os.makedirs('./output_image', exist_ok=True) 

DURATION_SEC = 4
WINDOW_SIZE = 4 # Ensure WINDOW_SIZE is divisible by DURATION_SEC
OUT_FPS = 1
MAX_TOKEN_PER_DURATION = 30
MAX_PIXELS = 360 * 640

def convert_qwen2_to_streaming(model: Qwen2VLForConditionalGeneration):
    # Patch vision encoder / blocks and decoder layers
    model.generate = MethodType(streaming_generate, model)
    model.prepare_inputs_for_generation = MethodType(prepare_multiturn_multimodal_inputs_for_generation, model)
    model._sample = MethodType(_sample, model)
    model.forward = MethodType(qwen2_vl_forward, model)
    model.model.forward = MethodType(model_forward, model.model)
    model.model.language_model.forward = MethodType(streaming_language_model_forward, model.model.language_model)
    model.model.visual.forward = MethodType(streaming_visual_encoder_forward, model.model.visual)
    for blk in model.model.visual.blocks:
        blk.forward = MethodType(streaming_visual_block_forward, blk)
        blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)
    for layer in model.model.language_model.layers:
        layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
        layer.self_attn.forward = MethodType(streaming_text_attn_forward, layer.self_attn)
    model.model.get_rope_index = MethodType(get_rope_index, model.model)
    return model 
