from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import itertools


@dataclass(slots=True)
class ConversationTurn:
    speaker: str
    text: str


@dataclass(slots=True)
class ConversationSample:
    turns: List[ConversationTurn]
    source: str = "synthetic"
    metadata: Optional[Dict[str, str]] = None

    def to_text(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)


def _parse_turn(raw: str) -> ConversationTurn:
    if ":" in raw:
        speaker, text = raw.split(":", 1)
        return ConversationTurn(speaker.strip(), text.strip())
    return ConversationTurn("Speaker", raw.strip())


def chunk_turns(items: Iterable[str], turns_per_sample: int) -> Iterator[List[str]]:
    iterator = iter(items)
    while True:
        chunk = list(itertools.islice(iterator, turns_per_sample))
        if not chunk:
            break
        yield chunk


def synthetic_samples(config: dict, turns_per_sample: int = 6) -> List[ConversationSample]:
    entries = config.get("data", {}).get("synthetic_turns", [])
    samples = []
    for idx, chunk in enumerate(chunk_turns(entries, turns_per_sample)):
        turns = [_parse_turn(item) for item in chunk]
        samples.append(ConversationSample(turns=turns, source=f"config_synthetic_{idx}"))
    return samples


def load_convbench(path: str | Path, limit: int = 1) -> List[ConversationSample]:
    import pandas as pd

    df = pd.read_excel(path)
    samples: List[ConversationSample] = []
    for _, row in df.head(limit).iterrows():
        turns = []
        for i in range(1, 4):
            prefix = ['first', 'second', 'third'][i-1]
            
            # Try different column naming conventions found in various versions of ConvBench
            instr = (
                row.get(f"instruction_{i}") or 
                row.get(f"The_{prefix}_turn_instruction") or
                row.get(f"The_{i}_turn_instruction")
            )
            
            ans = (
                row.get(f"reference_{i}") or 
                row.get(f"{prefix}_turn_answer")
            )
            
            if isinstance(instr, str) and instr.strip():
                turns.append(ConversationTurn("User", instr.strip()))
            if isinstance(ans, str) and ans.strip():
                turns.append(ConversationTurn("Assistant", ans.strip()))
        if turns:
            samples.append(ConversationSample(turns=turns, source="convbench"))
    return samples


def get_samples(config: dict, limit: int = 1) -> List[ConversationSample]:
    data_cfg = config.get("data", {})
    if data_cfg.get("convbench_path"):
        conv_path = Path(data_cfg["convbench_path"])
        if conv_path.exists():
            return load_convbench(conv_path, limit=limit)
    samples = synthetic_samples(config, turns_per_sample=data_cfg.get("turns_per_sample", 6))
    return samples[:limit]

