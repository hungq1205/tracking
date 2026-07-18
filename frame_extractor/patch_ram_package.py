"""
Patches the installed `ram` (recognize-anything, RAM++) package in place so
it imports under transformers==4.46.3 (see requirements.txt / CLAUDE.md's
"Frame Extractor" section for why that version is pinned in the first
place). `ram`'s BERT is a vendored copy of a much older transformers
internal API — this fixes the three specific breakages hit getting RAM++
running here, not a general compatibility shim.

Run once after installing frame_extractor/requirements.txt:
    conda activate hrtf
    python frame_extractor/patch_ram_package.py

Idempotent — safe to run again (each patch checks whether it already
applied before touching the file).
"""
import os

import ram

_RAM_DIR = os.path.dirname(ram.__file__)
_BERT_PY = os.path.join(_RAM_DIR, "models", "bert.py")
_UTILS_PY = os.path.join(_RAM_DIR, "models", "utils.py")

_BERT_OLD_IMPORT = """from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)"""

_BERT_NEW_IMPORT = '''from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """Removed from transformers>=5; only used by BertModel.prune_heads (not called
    during RAM/RAM++ inference), so a local copy of the old, unmodified utility is
    enough to keep this vendored BERT importable without pinning transformers."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index'''

_TOKENIZER_OLD = "    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]\n"
_TOKENIZER_NEW = (
    "    # additional_special_tokens_ids was removed from transformers>=5;\n"
    "    # convert_tokens_to_ids on the token we just added is equivalent.\n"
    "    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids('[ENC]')\n"
)


def _patch(path: str, old: str, new: str, label: str) -> None:
    src = open(path).read()
    if new in src:
        print(f"[patch_ram_package] {label}: already patched, skipping")
        return
    if old not in src:
        raise RuntimeError(
            f"[patch_ram_package] {label}: expected old text not found in {path} — "
            "the installed `ram` package version may differ from what this patch targets."
        )
    open(path, "w").write(src.replace(old, new))
    print(f"[patch_ram_package] {label}: patched")


if __name__ == "__main__":
    _patch(_BERT_PY, _BERT_OLD_IMPORT, _BERT_NEW_IMPORT, "bert.py imports")
    _patch(_UTILS_PY, _TOKENIZER_OLD, _TOKENIZER_NEW, "init_tokenizer")
