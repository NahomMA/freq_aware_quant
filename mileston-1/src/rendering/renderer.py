from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import textwrap

from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager

from src.data.conversations import ConversationSample, ConversationTurn


@dataclass(slots=True)
class RenderResult:
    path: Path
    width: int
    height: int
    text: str


class ConversationRenderer:
    def __init__(self, config: dict):
        self.font_cfg = config.get("font", {})
        self.render_cfg = config.get("rendering", {})

    def _load_font(self) -> ImageFont.FreeTypeFont:
        family = self.font_cfg.get("family", "Verdana.ttf")
        if family.endswith('.ttf'):
            family = family[:-4]
            
        weight = self.font_cfg.get("weight", "normal")
        size = self.font_cfg.get("size", 9)
        try:
            font_path = matplotlib.font_manager.findfont(
                matplotlib.font_manager.FontProperties(family=family, weight=weight)
            )
            return ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default()

    def render(self, conversation: ConversationSample, output_path: str | Path) -> RenderResult:
        font = self._load_font()
        
        margins = self.render_cfg.get("margins", [10, 10])
        padding_x = margins[0]
        padding_y = margins[1]
        line_height = self.render_cfg.get("line_spacing", 10)
        bg_color = self.render_cfg.get("background_color", "#FFFFFF")
        text_color = self.render_cfg.get("text_color", "#000000")
        
        # Use fixed width from config to force wrapping
        page_size = self.render_cfg.get("page_size", [1024, 2000])
        target_width = page_size[0]
        
        # Calculate max characters per line based on font width (approximate)
        avg_char_width = font.getbbox("x")[2]
        content_width = target_width - (2 * padding_x)
        chars_per_line = int(content_width / avg_char_width)

        lines = []
        for turn in conversation.turns:
            raw_line = f"{turn.speaker}: {turn.text}"
            wrapped = textwrap.wrap(raw_line, width=chars_per_line)
            lines.extend(wrapped)
            lines.append("") # Add spacing between turns

        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        total_text_height = 0
        line_metrics = []

        for line in lines:
            bbox = temp_draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            spacing = max(height, line_height)
            
            line_metrics.append((width, spacing))
            total_text_height += spacing

        img_width = target_width
        img_height = int(total_text_height + 2 * padding_y)
        
        # Ensure dimensions are even
        if img_width % 8 != 0: img_width += (8 - img_width % 8)
        if img_height % 8 != 0: img_height += (8 - img_height % 8)

        image = Image.new("RGB", (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(image)

        current_y = padding_y
        for i, line in enumerate(lines):
            _, spacing = line_metrics[i]
            draw.text((padding_x, current_y), line, fill=text_color, font=font)
            current_y += spacing

        output_path = Path(output_path)
        image.save(output_path)

        return RenderResult(
            path=output_path,
            width=img_width,
            height=img_height,
            text="\n".join(lines),
        )
