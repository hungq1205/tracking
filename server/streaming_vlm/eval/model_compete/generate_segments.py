import argparse
import json
from streaming_vlm.eval.model_compete.eval_livecc import eval_livecc
try:
    from streaming_vlm.eval.model_compete.eval_streaming_qwen import eval_streaming_qwen
    from streaming_vlm.inference.inference import load_model_and_processor
except:
    pass
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from streaming_vlm.data.lmm_dataset import get_phrase_before_timestamp
import os
from tqdm import tqdm
def load_eval_metadata(eval_metadata_path: str, start_line: int, end_line: int):
    data_list = []
    with open(eval_metadata_path, "r") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start_line:
                continue
            if line_idx >= end_line:
                break
            data = json.loads(line)
            data_list.append(data)
    return data_list
    
# --- Put at the top or __main__ front: minimum tool function ---
def latest_end_time(jsonl_path: str, video_name: str) -> int:
    """Scan jsonl, return the maximum end_time for this video; compatible with old format {video: [items]} and new format store by chunk."""
    latest = 0
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if video_name in obj and isinstance(obj[video_name], list) and obj[video_name]:
                    et = obj[video_name][-1].get("end_time", 0)
                    if isinstance(et, (int, float)):
                        latest = max(latest, int(et))
                elif obj.get("video") == video_name:
                    if isinstance(obj.get("chunk_end"), (int, float)):
                        latest = max(latest, int(obj["chunk_end"]))
                    else:
                        items = obj.get("items") or []
                        if items:
                            et = items[-1].get("end_time", 0)
                            if isinstance(et, (int, float)):
                                latest = max(latest, int(et))
    except FileNotFoundError:
        pass
    return latest

def append_chunk(jsonl_path: str, video_name: str, chunk_start: int, result: list):
    chunk_end = int(result[-1]["end_time"]) if result else int(chunk_start)
    obj = {
        "video": video_name,
        "chunk_start": int(chunk_start),
        "chunk_end": chunk_end,
        "items": result,
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="chenjoya/LiveCC-7B-Instruct")
    parser.add_argument("--model_base", type=str, default="LiveCC", choices=["Qwen","LiveCC"])
    parser.add_argument("--eval_metadata", type=str, default="")
    parser.add_argument("--start_time", type=int, default=1000)
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--start_line", type=int, default=0)
    parser.add_argument("--end_line", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="result.jsonl")
    parser.add_argument("--query", type=str, default="Please describe the video.")
    args = parser.parse_args()

    data_list = load_eval_metadata(args.eval_metadata, args.start_line, args.end_line)

    if args.model_base == "Qwen":
        model, processor = load_model_and_processor(args.model_path, "Qwen2_5")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map='cuda:0', attn_implementation='flash_attention_2'
        )
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

    for data_to_generate in tqdm(data_list):
        start_time = args.start_time
        end_time = int(data_to_generate['content'][-1][1])
        video_name = os.path.basename(data_to_generate['video'])

        last_time = latest_end_time(args.out_dir, video_name)
        if last_time >= end_time:
            print(f"skip {video_name}: already done (last_time={last_time} >= end_time={end_time})")
            continue
        if last_time > start_time:
            print(f"resume {video_name} from {last_time}s (skip first {last_time - start_time}s)")

        for chunk_start in range(max(start_time, last_time), end_time, args.duration):
            # Read latest progress again to avoid duplication
            upto = latest_end_time(args.out_dir, video_name)
            if upto > chunk_start:
                print(f"skip {video_name} chunk [{chunk_start}, {min(end_time, chunk_start+args.duration)}), already up to {upto}")
                continue

            duration = min(args.duration, end_time - chunk_start)
            if args.model_base == "Qwen":
                print("initializing Qwen model")
                result = eval_streaming_qwen(
                    model=model, processor=processor,
                    video_path=data_to_generate['video'],
                    previous_text=get_phrase_before_timestamp(data_to_generate['content'], chunk_start)[0],
                    start_time=chunk_start, temperature=args.temperature,
                    duration=duration, query=args.query
                )
            else:
                print("initializing LiveCC model")
                result = eval_livecc(
                    model=model, processor=processor,
                    video_path=data_to_generate['video'],
                    query=args.query,
                    start_time=chunk_start, duration=duration,
                    temperature=args.temperature
                )

            append_chunk(args.out_dir, video_name, chunk_start, result)
