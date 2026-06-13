import functools, torch, os, tqdm
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl() # important. our model is trained with this. keep consistency
from transformers import AutoProcessor, LogitsProcessor, logging
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2VLProcessor
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader
from qwen_vl_utils import process_vision_info
from transformers import set_seed
set_seed(42)

class LiveCCDemoInfer:
    fps = 2
    initial_fps_frames = 2
    streaming_fps_frames = 2
    initial_time_interval = initial_fps_frames / fps
    streaming_time_interval = streaming_fps_frames / fps
    frame_time_interval = 1 / fps
    def __init__(self, model = None, processor = None, model_path: str = None, device_id: int = 0):
        if model is None:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", 
                device_map=f'cuda:{device_id}', 
                attn_implementation='flash_attention_2'
            )
        else:
            self.model = model
        if processor is None:
            self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        else:
            self.processor = processor
        self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": 'livecc'},
            ]
        }
        texts = self.processor.apply_chat_template([message], tokenize=False) # '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nlivecc<|im_end|>\n'
        self.system_prompt_offset = texts.index('<|im_start|>user') 
        self._cached_video_readers_with_hw = {} 


    def live_cc(
        self,
        query: str,
        state: dict,
        max_pixels: int = 384 * 28 * 28,
        default_query: str = 'Please describe the video.',
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        start_time=0,
        temperature = 0.7,
        **kwargs,
    ): 
        """
        state: dict, (maybe) with keys:
            video_path: str, video path
            video_timestamp: float, current video timestamp
            last_timestamp: float, last processed video timestamp
            last_video_pts_index: int, last processed video frame index
            video_pts: np.ndarray, video pts
            last_history: list, last processed history
            past_key_values: llm past_key_values
            past_ids: past generated ids
        """

        # 1. preparation: video_reader, and last processing info
        
        video_timestamp, last_timestamp = state.get('video_timestamp', 0), state.get('last_timestamp', -1 / self.fps)
        video_path = state['video_path']
        if video_path not in self._cached_video_readers_with_hw:
            if not os.path.exists(video_path):
                video_path = os.path.join(os.environ['EVAL_DATASET_PATH'], video_path)
            self._cached_video_readers_with_hw[video_path] = get_smart_resized_video_reader(video_path, max_pixels)
            video_reader = self._cached_video_readers_with_hw[video_path][0]
            video_reader.get_frame_timestamp(0) # array([0.0, 0.01668333], dtype=float32)
            state['video_pts'] = torch.from_numpy(video_reader._frame_pts[:, 1]) # end time of each frame
            state['last_video_pts_index'] = -1
        video_pts = state['video_pts']
        if last_timestamp + self.frame_time_interval > video_pts[-1]:
            state['video_end'] = True
            return  
        video_reader, resized_height, resized_width = self._cached_video_readers_with_hw[video_path] # (<decord.video_reader.VideoReader object at 0x7f7257db6850>, 392, 728)
        last_video_pts_index = state['last_video_pts_index']

        # 2. which frames will be processed
        initialized = last_timestamp >= 0
        if not initialized:
            video_timestamp = max(video_timestamp, self.initial_time_interval)
        if video_timestamp <= last_timestamp + self.frame_time_interval:
            return
        # tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000])
        timestamps = torch.arange(last_timestamp + self.frame_time_interval, video_timestamp, self.frame_time_interval) # add compensation
        if not initialized:
            timestamps = timestamps + start_time

        # 3. fetch frames in required timestamps
        clip, clip_timestamps, clip_idxs = get_smart_resized_clip(video_reader, resized_height, resized_width, timestamps, video_pts, video_pts_index_from=last_video_pts_index+1)
        state['last_video_pts_index'] = clip_idxs[-1]
        state['last_timestamp'] = clip_timestamps[-1]
        # 4. organize to interleave frames
        interleave_clips, interleave_timestamps = [], [] # list of tensor
        if not initialized:
            interleave_clips.append(clip[:self.initial_fps_frames])
            interleave_timestamps.append(clip_timestamps[:self.initial_fps_frames])
            clip = clip[self.initial_fps_frames:]
            clip_timestamps = clip_timestamps[self.initial_fps_frames:]
        if len(clip) > 0:
            interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
            interleave_timestamps.extend(list(clip_timestamps.split(self.streaming_fps_frames)))

        # 5. make conversation and send to model
        for clip, timestamps in zip(interleave_clips, interleave_timestamps):
            start_timestamp, stop_timestamp = timestamps[0].item(), timestamps[-1].item() + self.frame_time_interval
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'Time={start_timestamp:.1f}-{stop_timestamp:.1f}s'},
                    {"type": "video", "video": clip}
                ]
            }
            if not query and not state.get('query', None):
                query = default_query
                print(f'No query provided, use default_query={default_query}')
            if query and state.get('query', None) != query:
                message['content'].append({"type": "text", "text": query})
                state['query'] = query
                previous = {'role': 'previous text', 'content': ''}
                message = [previous, message]
            

            texts = self.processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, return_tensors='pt') # '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nTime=0.0-3.0s<|vision_start|><|video_pad|><|vision_end|>Please describe the video.<|im_end|>\n<|im_start|>assistant\n'
            past_ids = state.get('past_ids', None)
            if past_ids is not None:
                texts = '<|im_end|>\n' + texts[self.system_prompt_offset:]
            inputs = self.processor(
                text=texts,
                images=None,
                videos=[clip],
                return_tensors="pt",
                return_attention_mask=False
            )
            inputs.to('cuda')
            if past_ids is not None:
                inputs['input_ids'] = torch.cat([past_ids, inputs.input_ids], dim=1) 
            if self.model.device == 'cpu':
                self.model.to("cuda")
            outputs = self.model.generate(
                **inputs, 
                past_key_values=state.get('past_key_values', None), 
                return_dict_in_generate=True, do_sample=do_sample, 
                repetition_penalty=repetition_penalty,
                max_new_tokens = 20,
                temperature=temperature,
                pad_token_id=self.model.config.eos_token_id,
            )
            state['past_key_values'] = outputs.past_key_values
            state['past_ids'] = outputs.sequences[:, :-1]
            yield (start_timestamp, stop_timestamp), self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True), state

def eval_livecc(model = None, processor = None, 
            model_path = "chenjoya/LiveCC-7B-Instruct", 
            video_path =  "/data/ruyi/dataset/streaming_vlm/Youtube_NBA/lKr5NocH5XM.mp4", 
            query = "Please describe the video.",
            previous_text = "",
            start_time:float = 30, 
            duration:int = 100,
            temperature = 0.7):
            
    infer = LiveCCDemoInfer(model=model, processor=processor, model_path=model_path)
    state = {'video_path': video_path}
    commentaries = []
    for t in range(start_time, start_time + duration):
        state['video_timestamp'] = t
        for (start_t, stop_t), response, state in infer.live_cc(
            query=query, state=state, 
            max_pixels = 384 * 28 * 28, repetition_penalty=1.05, 
            streaming_eos_base_threshold=0.0, streaming_eos_threshold_step=0,
            start_time = start_time, # FIXME 检查这个实现是否正确
            temperature = temperature,
        ):
            print(f't={t}s, {start_t}s-{stop_t}s: {response}')
            commentaries.append({'response':response[:-4], 'start_time':int(start_t), 'end_time':int(stop_t)})
            if stop_t >= start_time + duration:
                del infer
                torch.cuda.empty_cache()
                return commentaries
        if state.get('video_end', False):
            break
    return commentaries

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='chenjoya/LiveCC-7B-Instruct')
    parser.add_argument('--video_path', type=str, default='/data/ruyi/dataset/streaming_vlm/Youtube_NBA/lKr5NocH5XM.mp4')
    parser.add_argument('--query', type=str, default='Please describe the video.')
    parser.add_argument('--start_time', type=int, default=1300)
    parser.add_argument('--duration', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype="auto", 
                device_map=f'cuda:0', 
                attn_implementation='flash_attention_2'
            )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

    for i in range(10):
        print(eval_livecc(model=model, processor=processor, video_path=args.video_path, query=args.query, start_time=args.start_time+i*args.duration, duration=args.duration, temperature=args.temperature))

