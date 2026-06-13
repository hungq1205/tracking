from transformers import AutoTokenizer, AutoModelForCausalLM
from streaming_vlm.inference.inference import streaming_inference

def eval_streaming_qwen(model = None, processor = None, 
            model_path = "xrorrim/checkpoint-150-model_-20250824-203648", 
            video_path =  "/data/ruyi/dataset/streaming_vlm/Youtube_Basketball/lKr5NocH5XM.mp4", 
            query = "Please describe the video.",
            previous_text = "",
            start_time:float = 30, 
            duration:int = 10,
            temperature = 0.7,
            previous_sink = 512,
            previous_sliding_window = 512,
            ):
    
    return streaming_inference(model_path, 
                        video_path, 
                        model = model,
                        processor = processor,
                        previous_text = previous_text, 
                        skip_first_chunk = start_time,
                        previous_sink = previous_sink,
                        previous_sliding_window = previous_sliding_window,
                        temperature = temperature,
                        duration = duration,
                        query = query,
                        )
if __name__ == "__main__":
    print(eval_streaming_qwen())