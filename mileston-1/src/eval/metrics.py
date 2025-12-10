from __future__ import annotations

import math
from typing import Tuple

def word_recall(reference: str, hypothesis: str) -> float:
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    if not ref_words:
        return 0.0
    return len(ref_words & hyp_words) / len(ref_words)


def token_compression_ratio(text_token_count: int, visual_tokens: int) -> float:
    if visual_tokens == 0:
        return math.inf
    return text_token_count / visual_tokens

