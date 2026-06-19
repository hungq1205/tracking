from streaming_vlm.inference.qwen3.vision_forward import (
    streaming_visual_attention_forward,
    streaming_visual_block_forward,
    streaming_visual_encoder_forward,
)
from streaming_vlm.inference.qwen3.language_forward import (
    streaming_language_model_forward,
    streaming_text_flash_attn_forward,
    streaming_text_decoder_layer_forward,
    _update_causal_mask,
)
from streaming_vlm.inference.qwen3.model_forward import (
    model_forward,
    qwen3_vl_forward,
    prepare_inputs_for_streaming_generation,
)
from streaming_vlm.inference.qwen3.pos_emb import get_rope_index
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from types import MethodType
from streaming_vlm.inference.generate.streaming_generate_qwen import streaming_generate, _sample
from streaming_vlm.inference.generate.prepare_generation import prepare_multiturn_multimodal_inputs_for_generation
import os
os.makedirs('./output_image', exist_ok=True)


def convert_qwen3_to_streaming(model: Qwen3VLForConditionalGeneration):
    model.generate = MethodType(streaming_generate, model)
    model.prepare_inputs_for_generation = MethodType(prepare_multiturn_multimodal_inputs_for_generation, model)
    model._sample = MethodType(_sample, model)

    model.forward = MethodType(qwen3_vl_forward, model)
    model.model.forward = MethodType(model_forward, model.model)
    model.model.language_model.forward = MethodType(streaming_language_model_forward, model.model.language_model)
    model.model.language_model._update_causal_mask = MethodType(_update_causal_mask, model.model.language_model)
    for layer in model.model.language_model.layers:
        layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
        layer.self_attn.forward = MethodType(streaming_text_flash_attn_forward, layer.self_attn)

    model.model.visual.forward = MethodType(streaming_visual_encoder_forward, model.model.visual)
    for blk in model.model.visual.blocks:
        blk.forward = MethodType(streaming_visual_block_forward, blk)
        blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)

    model.model.get_rope_index = MethodType(get_rope_index, model.model)
    return model
