class StreamingArgs:
    def __init__(self, pos_mode: str, all_text: bool = False):
        self.pos_mode = pos_mode
        self.all_text = all_text
        assert pos_mode in ["append", "shrink"], "pos_mode must be in ['append', 'shrink']"
        # append mode grows indefinitely
        # shrink mode ensures current kv cache position ids are continuous
        self.input_ids = None    # shrink mode needs complete input ids passed to attention to detect video frames    
        self.video_grid_thw = None
        self.second_per_grid_ts = None
        