from streaming_vlm.inference.inference import streaming_inference
import argparse
import os, json
from datetime import datetime
from streaming_vlm.inference.inference import DEFAULT_WINDOW_SIZE, DEFAULT_CHUNK_DURATION, DEFAULT_TEXT_ROUND, DEFAULT_TEMPERATURE, DEFAULT_TEXT_SINK, DEFAULT_TEXT_SLIDING_WINDOW
import torch
# (a) FullAttention in paper
baseline_a_config = {
    "window_size": 100000,
    "chunk_duration": DEFAULT_CHUNK_DURATION,
    "text_round": 100000,
    "text_sink": None,
    "text_sliding_window": None,
    "recompute": False,
}
# (b) sliding window w/o overlapping in paper
baseline_b_config = {
    "window_size": 100,
    "chunk_duration": DEFAULT_CHUNK_DURATION,
    "text_round": 100,
    "text_sink": None,
    "text_sliding_window": None,
    "recompute": False,
}
# (c) Sliding window w/ overlapping in paper
baseline_c_config = {
    "window_size": DEFAULT_WINDOW_SIZE,
    "chunk_duration": DEFAULT_CHUNK_DURATION,
    "text_round": DEFAULT_TEXT_ROUND,
    "text_sink": None,
    "text_sliding_window": None,
    "recompute": True,
}
# (d) StreamingVLM in paper
streaming_vlm_config = {
    "window_size": DEFAULT_WINDOW_SIZE,
    "chunk_duration": DEFAULT_CHUNK_DURATION,
    "text_round": DEFAULT_TEXT_ROUND,
    "text_sink": DEFAULT_TEXT_SINK,
    "text_sliding_window": DEFAULT_TEXT_SLIDING_WINDOW,
    "recompute": False,
}

if __name__ == "__main__":
    default_video = "Baidu_FIFA/FIFA_Worldcup_2018年06月23日 2018年俄罗斯世界杯F组第2轮 韩国VS墨西哥 1080I ITV 英语 2nd.ts.mp4"
    args = argparse.ArgumentParser()
    args.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"])
    args.add_argument("--all_text", action="store_true", default=False)
    args.add_argument("--model_path", type=str, default="mit-han-lab/StreamingVLM")
    args.add_argument("--model_base", type=str, choices=["Qwen2_5", "Qwen2"], default="Qwen2_5")
    args.add_argument("--video_path", type=str, default=default_video)
    args.add_argument("--previous_text", type=str, default="This is a NBA game between the Golden State Warriors and the Detroit Pistons")
    args.add_argument("--skip_first_chunk", type=int, default=0)
    args.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    args.add_argument("--output_dir", type=str)

    args.add_argument("--test_data_json", type=str, default=None)
    args.add_argument("--test_data_idx", type=int, default=None)

    args.add_argument("--baseline_mode", type=str, default="d", choices=["a", "b", "c", "d"])
    
    args = args.parse_args()

    print(f'inferencing model {args.model_path}')
    baseline_mode = args.baseline_mode
    config_used = baseline_a_config if baseline_mode == "a" else baseline_b_config if baseline_mode == "b" else baseline_c_config if baseline_mode == "c" else streaming_vlm_config
    del args.baseline_mode
    duration = 1000
    chunk = 1000
    time_results, token_decoded_nums =[], []
    for i in range(0, duration, chunk):
        args.skip_first_chunk = i
        time_result, token_decoded_num = streaming_inference(**args.__dict__, **config_used, duration=chunk, time_test=True, quiet=False)
        
        import gc, torch; torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        time_results.extend(time_result)
        token_decoded_nums.extend(token_decoded_num)
    
    def _safe(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    chunk_dur = config_used["chunk_duration"]
    num_chunks = len(time_results)
    records = []
    for i, sec_time in enumerate(time_results):
        gen_t = float(sec_time.get("GEN", 0.0))
        dec = int(token_decoded_nums[i]) if i < len(token_decoded_nums) else 0
        t_start = (i + args.skip_first_chunk) * chunk_dur
        video_len = (i + 1) * chunk_dur  # Cumulative video length
        records.append({
            "chunk_index": i,
            "time_start_sec": t_start,
            "video_len_sec": video_len,
            "gen_time_sec": gen_t,
            "decoded_tokens": dec,
            "gen_time_per_token": (gen_t / dec) if dec > 0 else None
        })
    meta = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model_path": args.model_path,
        "model_base": args.model_base,
        "video_path": args.video_path,
        "pos_mode": args.pos_mode,
        "all_text": args.all_text,
        "skip_first_chunk": args.skip_first_chunk,
        "temperature": args.temperature,
        "mode": "baseline_a" if baseline_mode == "a" else "baseline_b" if baseline_mode == "b" else "baseline_c" if baseline_mode == "c" else "streaming",
        "window_size": config_used["window_size"],
        "chunk_duration": config_used["chunk_duration"],
        "text_round": config_used["text_round"],
        "text_sink": config_used["text_sink"],
        "text_sliding_window": config_used["text_sliding_window"],
        "recompute": config_used["recompute"],
        "duration_tested_sec": duration
    }
    out_dir = os.path.join("output", "efficiency")
    os.makedirs(out_dir, exist_ok=True)
    auto_name = f"{_safe(meta['mode'])}__{_safe(meta['model_base'])}__{_safe(meta['model_path'])}__{_safe(os.environ.get('QWENVL_FPS', '2.0'))}___{_safe(os.path.basename(meta['video_path']))}__s{meta['skip_first_chunk']}__w{meta['window_size']}__c{meta['chunk_duration']}__t{meta['text_round']}__{meta['timestamp']}.json"
    result_path = os.path.join(out_dir, auto_name)
    payload = {
        "meta": meta,
        "per_chunk": records,
        "summary": {
            "num_chunks": num_chunks,
            "avg_gen_time_sec": float(sum(r["gen_time_sec"] for r in records) / max(num_chunks, 1)),
            "avg_gen_time_per_token": float(
                sum((r["gen_time_per_token"] or 0.0) for r in records if r["gen_time_per_token"] is not None)
                / max(len([r for r in records if r["gen_time_per_token"] is not None]), 1)
            )
        }
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved efficiency json -> {result_path}")
    print(token_decoded_nums)
