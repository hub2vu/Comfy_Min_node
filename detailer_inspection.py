import json
from pathlib import Path
import re
import uuid

import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from server import PromptServer
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
                raise ValueError("Use all, none, indices such as 0,2, or image:index such as 0:1,1:0. Ranges such as 0-3 are supported.")
            image, start, end = match.groups()
            start, end = int(start), int(end) if end is not None else int(start)
            self.ranges.append((int(image) if image is not None else None, min(start, end), max(start, end)))

    def includes(self, image_index, mask_index):
        if self.text in ("", "all"):
            return True
        return any((image is None or image == image_index) and start <= mask_index <= end for image, start, end in self.ranges)


def _pil(value):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8))


def _tensor(image):
    return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255).unsqueeze(0)


def _font(size):
    for path in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _geometry(segment):
    return tuple(segment.bbox), tuple(segment.crop_region), str(segment.label)


def _region(segment, image_index, index):
    return {
        "id": f"{image_index}:{index}", "image_index": image_index, "index": index,
        "class_label": str(segment.label), "confidence": float(segment.confidence),
        "bbox": [float(value) for value in segment.bbox],
        "crop_region": [int(value) for value in segment.crop_region],
    }


class DetailerInspection:
    def __init__(self, run):
        self.run = run
        self.nodes = {}
        self.folder = Path(folder_paths.get_temp_directory()) / "min_detailer" / uuid.uuid4().hex
        self.views = {stage: {"stage": stage, "run_id": self.folder.name, "complete": False, "node_id": None, "rows": [], "overviews": []} for stage in ("detect", "mask", "crop", "regenerate", "composite")}
        self.counter = 0
        self.batch_index = 0
        self.mask_rows = {}
        self.processing_rows = {}
        self.output_ids = {}
        self.detected_segments = {}
        self.current = None
        self.cropped = None
        self.selection = RegionSelection()
        self.preview_only = False
        self.current_selected = True
        self.last_annotated = None
        self.last_combined_mask = None
        self.last_region_masks = []
        self.last_mask_overlay = None

    def bind(self, stage, node_id):
        if node_id is not None and stage not in self.nodes:
            self.nodes[stage] = str(node_id)
            self.views[stage]["node_id"] = str(node_id)

    def save(self, image):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        filename = f"{self.counter:05}.png"
        image.save(self.folder / filename, compress_level=1)
        return {"filename": filename, "subfolder": f"min_detailer/{self.folder.name}", "type": "temp"}

    def publish(self, stage):
        server = getattr(PromptServer, "instance", None)
        if server is not None and stage in self.nodes:
            server.send_sync("min-node.detailer-preview", json.loads(json.dumps(self.views[stage])), server.client_id)

    def ui(self, stage):
        return {"min_detailer": [self.views[stage]]}

    def report(self, stage=None):
        return json.dumps(self.views[stage] if stage else self.views, ensure_ascii=False, indent=2)

    def configure(self, selected_indices, preview_only):
        self.selection = RegionSelection(selected_indices)
        self.preview_only = preview_only
        for row in self.processing_rows.values():
            row["selected"] = self.selection.includes(row["image_index"], row["index"])
            row["will_process"] = not preview_only and row["selected"]
        self.current_selected = self.current is not None and not preview_only and self.selection.includes(self.current["image_index"], self.current["index"])
        if self.current is not None:
            if preview_only:
                self.current["status"] = "preview only"
            elif not self.current_selected:
                self.current["status"] = "not selected"
        self.views["regenerate"].update(selected_indices=selected_indices, preview_only=preview_only)

    def observe(self, phase):
        frames = self.run.frames()
        outer, face, detail = (frames.get(name, {}) for name in ("doit", "enhance_face", "do_detail"))
        self.batch_index = int(outer.get("i", self.batch_index))
        if phase == "detected":
            self._detect(face["image"], self.run.value)
        elif phase == "masked":
            self._mask(face["image"], self.run.value)
        elif phase == "before_crop":
            segment = detail["seg"]
            index = next(i for i, item in enumerate(detail["segs"][1]) if item is segment)
            self.current = self.processing_rows[self.batch_index, index]
            self.current.update(processing_order=int(detail["i"]), status="cropping", bbox=[float(x) for x in segment.bbox], crop_region=[int(x) for x in segment.crop_region])
            self.cropped = None
        elif phase == "crop_input":
            self.cropped = self.run.value.detach().clone()
            self.current["cropped"] = self.save(_pil(self.cropped))
            self.current["crop_size"] = list(self.cropped.shape[1:3][::-1])
            if not np.any(detail["seg"].cropped_mask):
                self.current["status"] = "empty mask"
        elif phase == "region_ready":
            self.current["seed"] = int(detail["seg_seed"])
            wildcard = detail.get("wildcard_item")
            if wildcard and wildcard.strip() in ("[SKIP]", "[STOP]"):
                self.current["status"] = wildcard.strip()
        elif phase == "cropped":
            self.current["upscaled"] = self.save(_pil(self.run.value))
            self.current["upscaled_size"] = list(self.run.value.shape[1:3][::-1])
            self.current["status"] = "ready"
            self.publish("crop")
        elif phase == "regenerated":
            if self.current_selected and self.run.value is not None:
                self.current["refined"] = self.save(_pil(self.run.value))
                self.current["status"] = "regenerated"
            elif self.current_selected:
                self.current["status"] = "skipped by original settings"
            self.publish("regenerate")
        elif phase == "composited":
            if self.current.get("refined"):
                self.current["status"] = "composited"
            if detail["enhanced_image"] is not None:
                self.current["processed"] = True
                self.output_ids[self.current["id"]] = (id(detail["enhanced_image"]), id(detail["enhanced_image_alpha"]))
        elif phase == "complete":
            crops = {id(image): index for index, image in enumerate(self.run.result[1])}
            alphas = {id(image): index for index, image in enumerate(self.run.result[2])}
            for row in self.processing_rows.values():
                if row["id"] in self.output_ids:
                    crop_id, alpha_id = self.output_ids[row["id"]]
                    row["final_crop_index"] = crops[crop_id]
                    row["final_alpha_index"] = alphas[alpha_id]
                    row["final_refined"] = self.save(_pil(self.run.result[1][crops[crop_id]]))
            self.views["composite"]["views"] = {stage: view for stage, view in self.views.items() if stage != "composite"}
            self.views["composite"]["rows"] = self.views["regenerate"]["rows"]
            for stage in self.views:
                self.views[stage]["complete"] = True
                self.publish(stage)

    def _detect(self, image, segs):
        self.detected_segments[self.batch_index] = list(segs[1])
        rows = [_region(segment, self.batch_index, index) for index, segment in enumerate(segs[1])]
        self.views["detect"]["rows"].extend(rows)
        preview = _pil(image).convert("RGB")
        draw = ImageDraw.Draw(preview)
        font = _font(max(12, min(24, preview.width // 35)))
        colors = ("#ffb347", "#5eead4", "#f472b6", "#a5b4fc", "#bef264")
        for row in rows:
            color = colors[row["index"] % len(colors)]
            draw.rectangle(row["crop_region"], outline=color, width=1)
            draw.rectangle(row["bbox"], outline=color, width=3)
            label = f"{row['id']}  {row['class_label']}  {row['confidence']:.3f}"
            x, y = row["bbox"][:2]
            text_box = draw.textbbox((x, max(0, y - 20)), label, font=font)
            draw.rectangle(text_box, fill="#151515")
            draw.text((x, max(0, y - 20)), label, font=font, fill=color)
        self.last_annotated = _tensor(preview)
        self.views["detect"]["overviews"].append({"image_index": self.batch_index, "image": self.save(preview)})
        self.publish("detect")

    def _mask(self, image, segs):
        h, w = segs[0]
        combined = np.zeros((h, w), dtype=np.uint8)
        self.last_region_masks = []
        rows = []
        available = list(enumerate(self.detected_segments[self.batch_index]))
        kind = "SAM refined" if self.run.parameters.get("sam_model_opt") is not None else "segmentation refined" if self.run.parameters.get("segm_detector_opt") is not None else "bbox detector mask"
        for index, segment in enumerate(segs[1]):
            row = _region(segment, self.batch_index, index)
            source = next((i for i, original in available if _geometry(original) == _geometry(segment)), None)
            if source is not None:
                available = [(i, item) for i, item in available if i != source]
            row.update(source_detection_index=source, mask_kind=kind)
            x1, y1, x2, y2 = segment.crop_region
            mask = np.asarray(segment.cropped_mask, dtype=np.float32)
            full = np.zeros((h, w), dtype=np.float32)
            full[y1:y2, x1:x2] = mask
            combined[y1:y2, x1:x2] |= (mask * 255).astype(np.uint8)
            self.last_region_masks.append(torch.from_numpy(full).unsqueeze(0))
            row["mask"] = self.save(_pil(full))
            row["region_preview"] = self.save(_pil(image[:, y1:y2, x1:x2, :]))
            rows.append(row)
            selected = self.selection.includes(self.batch_index, index)
            processing = {**row, "status": "not reached", "selected": selected,
                          "will_process": not self.preview_only and selected, "processed": False}
            self.processing_rows[self.batch_index, index] = processing
            self.views["crop"]["rows"].append(processing)
            self.views["regenerate"]["rows"].append(processing)
        self.mask_rows[self.batch_index] = rows
        self.views["mask"]["rows"].extend(rows)
        self.last_combined_mask = torch.from_numpy(combined.astype(np.float32) / 255).unsqueeze(0)
        overlay = _pil(image).convert("RGB")
        tint = Image.new("RGB", overlay.size, "#45d6c0")
        alpha = Image.fromarray((combined.astype(np.float32) * 0.45).astype(np.uint8))
        overlay.paste(tint, (0, 0), alpha)
        self.last_mask_overlay = _tensor(overlay)
        self.views["mask"]["overviews"].append({
            "image_index": self.batch_index, "image": self.save(_pil(self.last_combined_mask[0])), "overlay": self.save(overlay), "mask_kind": kind,
        })
        self.publish("mask")
        self.publish("crop")
        self.publish("regenerate")
