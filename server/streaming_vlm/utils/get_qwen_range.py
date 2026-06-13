SYSTEM_PROMPT_OFFSET = 58
TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "user": 872,
    "assistant": 77091,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|video_pad|>": 151656,
    "\n":198,
    'previous text':[19702, 1467],
    'Time':1462,
}

def get_qwen_range(input_ids, label:str , index, contain_lf=True):
    """
    Parameters
    -------
    input_ids : List[int]       ID sequence of length [seqlen]
    label     : str             Values in {'user','assistant','vision','user_text'}
    index     : int             The index-th segment (starting from 0)

    Returns
    -------
    start_idx, end_idx : Tuple[int, int]
        (start position, end position) of the index-th segment corresponding to label
    """
    # Find all start_indexes and end_indexes according to a pattern, with ... being free content in between
    # <|im_start|>user\n ...<|im_end|>
    # <|im_start|>assistant\n ...<|im_end|>
    # <|vision_start|><|video_pad|> ...<|vision_end|>  (here .. is <|video_pad|>)
    # <|vision_end|> ...<|im_end|>

    assert label in ['user', 'previous text', 'assistant', 'vision', 'user_text']
    input_ids = input_ids.flatten().tolist()

    # Prepare token-id level patterns for each label type
    if label == 'user':
        start_pat = [
            TOKEN_IDS["<|im_start|>"],
            TOKEN_IDS["user"],
        ]
        end_pat = [TOKEN_IDS["<|im_end|>"]]
    elif label == 'previous text':
        start_pat = [TOKEN_IDS["<|im_start|>"]] +  TOKEN_IDS["previous text"] + [TOKEN_IDS["\n"]]
        end_pat = [TOKEN_IDS["<|im_end|>"]]
    elif label == 'user_text':
        start_pat = [TOKEN_IDS["Time"]]
        end_pat = [TOKEN_IDS["<|vision_start|>"]]
    elif label == 'assistant':
        start_pat = [
            TOKEN_IDS["<|im_start|>"],
            TOKEN_IDS["assistant"],
        ]
        end_pat = [TOKEN_IDS["<|im_end|>"]]
    elif label == 'vision':
        start_pat = [TOKEN_IDS["<|vision_start|>"]]
        end_pat = [TOKEN_IDS["<|vision_end|>"]]

    n = len(input_ids)
    segments = []

    i = 0
    while i <= n - len(start_pat):
        # Check if start pattern matches
        if input_ids[i : i + len(start_pat)] == start_pat:
            start_idx = i
            j = i + len(start_pat)
            # Continue looking right for end pattern
            while j <= n - len(end_pat):
                if input_ids[j : j + len(end_pat)] == end_pat:
                    if j + len(end_pat) < n and input_ids[j + len(end_pat)] == TOKEN_IDS["\n"] and contain_lf: # In non-final rounds, there is a \n ending
                        segments.append((start_idx, j+len(end_pat)))
                    else:
                        segments.append((start_idx, j+len(end_pat)-1))
                    i = j + len(end_pat)  # Skip this segment, continue scanning forward
                    break
                j += 1
            else:
                break
        else:
            i += 1

    if label == 'user_text':
        return segments[index][0], segments[index][1]-1 # Because text uses the following vision start as marker
    return segments[index]
