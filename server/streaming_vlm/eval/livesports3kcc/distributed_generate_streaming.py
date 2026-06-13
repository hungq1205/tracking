import os
import json
import tqdm
import shutil
import argparse
import multiprocessing
from functools import partial
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from streaming_vlm.inference.inference import streaming_inference
from datasets import load_dataset
from baselines.livecc.utils.multiprocessor import local_mp

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed LiveCC generation over the LiveSports-3K CC split"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="HuggingFace model path, e.g., chenjoya/LiveCC-7B-Instruct"
    )
    parser.add_argument(
        "--not_instruct_model", action="store_true", dest="not_instruct_model", help="Disable instruct model mode"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel processes/gpus to use"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.15,
        help="Repetition penalty for generation. When performing livecc, 1.15 can remove most repetition."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="evaluation/livesports3kcc/livecc",
        help="Directory to write generated JSON outputs"
    )
    parser.add_argument(
        "--temperature", type=float,
        default=0.9,
        help="Temperature for generation"
    )
    return parser.parse_args()

def streaming_worker(
    device_id: int,
    model_name_or_path: str,
    save_dir: str,
    simple_ctx: bool,
    repetition_penalty: float,
    num_workers: int,
    temperature: float,
):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype="auto", 
        device_map=f'cuda:{device_id}', 
        attn_implementation='flash_attention_2'
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)

    ds = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test")
    idxs = list(range(len(ds)))
    idxs_on_device = idxs[device_id::num_workers]

    # Prepare temporary save folder for this model
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm.tqdm(idxs_on_device, desc=f"Device {device_id}", total=len(idxs_on_device)):
        save_path = os.path.join(save_dir, f"{idx}.json")
        if os.path.exists(save_path):
            continue

        record = ds[idx]
        video = record.get("video")
        if not os.path.exists(video):
            video = os.path.join(os.getenv("LIVESPORTS3K_PATH"), video)
        video_id = record.get("video_id")
        event_id = record.get("event_id")
        video_start = record.get("begin")
        video_end = record.get("end")
        title = record.get("event_title")
        preasr = record.get("preasr_text")

        if simple_ctx:
            title = '' if preasr else title # title or preasr
            overall_prompt = f'{title}\n{preasr}'.strip()
        else:
            commentary_prompt = (
                "You are an expert video commentator providing real-time, insightful, "
                "and engaging commentary on visual content.\n"
            )
            overall_prompt = commentary_prompt
            if title:
                overall_prompt += f"This is a video titled \"{title}\".\n"
        print(f"video: {video}")
        print(f"overall_prompt: {overall_prompt}")
        print(f"preasr: {preasr}")
        print(f"video_start: {video_start}")
        print(f"duration: {video_end-video_start}")

        responses = streaming_inference(
            model=model,
            processor=processor,
            query=overall_prompt,
            previous_text=preasr if preasr else "",
            video_path=video, skip_first_chunk=video_start, duration=video_end-video_start,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            quiet=True,
        )
        overall_cc = "".join(response['response'] for response in responses)


        print("--------------------------------")
        print(overall_cc)
        print("--------------------------------")
        with open(save_path, 'w') as wf:
            json.dump({
                "video_id": video_id,
                'event_id': event_id,
                "begin": video_start,
                "end": video_end,
                "pred": overall_cc
            }, wf)

if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    save_dir = os.path.join(args.output_dir, os.path.basename(args.model_name_or_path))
    worker_fn = partial(
        streaming_worker,
        model_name_or_path=args.model_name_or_path,
        save_dir=save_dir,
        simple_ctx=args.not_instruct_model,
        repetition_penalty=args.repetition_penalty,
        num_workers=args.num_workers,
        temperature=args.temperature,
    )
    local_mp(
        list(range(args.num_workers)),
        worker_fn,
        desc="livecc_generation",
        num_workers=args.num_workers
    )
    # jsons -> jsonl
    save_path = save_dir + '.jsonl'
    with open(save_path, 'w') as wf:
        for file in os.listdir(save_dir):
            try:
                datum = json.load(open(os.path.join(save_dir, file))) 
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            wf.write(json.dumps(datum) + '\n')
    # remove save_dir
    shutil.rmtree(save_dir)

