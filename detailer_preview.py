import json
from pathlib import Path
import re
import uuid

import folder_paths
from comfy_execution.utils import get_executing_context
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


class RegionSelection:
    def __init__(self, text="all"):
        self.text = text.strip().lower()
        self.ranges = []
        if self.text in ("", "all", "none"):
            return
        for token in re.split(r"[,;\s]+", self.text):
            match = re.fullmatch(r"(?:(\d+):)?(\d+)(?:-(\d+))?", token)
            if match is None:
                raise ValueError("Use all, none, indices such as 0,2, or image:index such as 0:1,1:0.")
            image, first, last = match.groups()
            first, last = int(first), int(last) if last is not None else int(first)
            self.ranges.append((int(image) if image is not None else None, min(first, last), max(first, last)))

    def includes(self, image, index):
        return self.text in ("", "all") or any((batch is None or image == batch) and first <= index <= last for batch, first, last in self.ranges)


def _pil(value, mask=False):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if mask:
        array = array.reshape(-1, *array.shape[-2:])[0]
    elif array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))


def _tensor(image):
    return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).unsqueeze(0)


def annotate(image, rows):
    marked = _pil(image).convert("RGB")
    draw = ImageDraw.Draw(marked)
    font = ImageFont.load_default()
    for path in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            font = ImageFont.truetype(path, max(12, min(24, marked.width // 35)))
            break
    colors = ("#ffb347", "#5eead4", "#f472b6", "#a5b4fc")
    for row in rows:
        color = colors[row["index"] % len(colors)]
        draw.rectangle(row["crop_region"], outline=color, width=1)
        draw.rectangle(row["bbox"], outline=color, width=3)
        x, y = row["bbox"][:2]
        text = f"{row['id']}  {row['class_label']}  {row['confidence']:.3f}"
        position = x, max(0, y - 20)
        draw.rectangle(draw.textbbox(position, text, font=font), fill="#151515")
        draw.text(position, text, font=font, fill=color)
    return _tensor(marked)


def mask_overlay(image, mask):
    original = _pil(image).convert("RGB")
    tint = Image.new("RGB", original.size, "#45d6c0")
    alpha = Image.fromarray((mask[0].cpu().numpy() * 255 * 0.45).astype(np.uint8))
    original.paste(tint, (0, 0), alpha)
    return _tensor(original)


def preview_ui(cls, stage, rows, overview=None, overlay=None, image_index=0):
    folder = Path(folder_paths.get_temp_directory()) / "min_detailer" / uuid.uuid4().hex
    counter = 0

    def save(value, mask=False):
        nonlocal counter
        folder.mkdir(parents=True, exist_ok=True)
        counter += 1
        filename = f"{counter:05}.png"
        _pil(value, mask).save(folder / filename, compress_level=1)
        return {"filename": filename, "subfolder": f"min_detailer/{folder.name}", "type": "temp"}

    rendered = []
    for row in rows:
        row = row.copy()
        for key in ("cropped", "upscaled", "refined", "final_refined", "region_preview", "mask"):
            if key in row:
                row[key] = save(row[key], key == "mask")
        rendered.append(row)
    node_id = cls.hidden.unique_id if cls.hidden else None
    if node_id is not None and cls.hidden.dynprompt is not None:
        node_id = cls.hidden.dynprompt.get_display_node_id(node_id)
    context = get_executing_context()
    view = {"stage": stage, "run_id": context.prompt_id if context else folder.name,
            "node_id": str(node_id) if node_id is not None else None, "rows": rendered, "overviews": [], "complete": True}
    if overview is not None:
        entry = {"image_index": image_index, "image": save(overview, stage == "mask")}
        if overlay is not None:
            entry["overlay"] = save(overlay)
        view["overviews"].append(entry)
    return {"min_detailer": [view]}, json.dumps(view, ensure_ascii=False, indent=2)
