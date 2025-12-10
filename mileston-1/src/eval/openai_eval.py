from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.eval.metrics import word_recall

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


@dataclass(slots=True)
class VLMResult:
    success: bool
    text: str
    recall: float
    error: Optional[str] = None


load_dotenv()


class OpenAIImageEvaluator:
    def __init__(self, config: dict):
        self.cfg = config.get("evaluation", {})
        self.model = self.cfg.get("openai_model", "gpt-4o")
        self.prompt = self.cfg.get("prompt", "Describe this image.")
        self.max_tokens = self.cfg.get("max_tokens", 512)
        self.temperature = self.cfg.get("temperature", 0.0)

    def _client(self):
        if OpenAI is None:
            raise RuntimeError("openai package not installed.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        return OpenAI(api_key=api_key)

    def _encode_image(self, path: Path) -> str:
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def evaluate(self, image_path: str | Path, reference_text: str) -> VLMResult:
        image_path = Path(image_path)
        try:
            client = self._client()
        except (RuntimeError, ValueError) as exc:
            return VLMResult(success=False, text="", recall=0.0, error=str(exc))

        payload = self._encode_image(image_path)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  
            return VLMResult(success=False, text="", recall=0.0, error=str(exc))

        text = response.choices[0].message.content or ""
        recall = word_recall(reference_text, text)
        return VLMResult(success=True, text=text, recall=recall)

