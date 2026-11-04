import importlib
import math
from types import SimpleNamespace

import comfy.samplers
import nodes
import numpy as np
import torch
from comfy_api.latest import io

from .detailer_preview import RegionSelection, annotate, mask_overlay, preview_ui


# Parallel standard lists. No region/session object is passed between nodes.
REGION_FIELDS = [
    ("region_index", io.Int), ("labels", io.String), ("confidences", io.Float), ("bbox_dtype", io.String),
    ("bbox_x1", io.Float), ("bbox_y1", io.Float), ("bbox_x2", io.Float), ("bbox_y2", io.Float),
    ("crop_x1", io.Int), ("crop_y1", io.Int), ("crop_x2", io.Int), ("crop_y2", io.Int),
    ("masks", io.Mask),
]
BBOX = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
CROP = ("crop_x1", "crop_y1", "crop_x2", "crop_y2")
HIDDEN = [io.Hidden.unique_id, io.Hidden.dynprompt]


def _impact(module):
    return importlib.import_module(f"impact.{module}")


def _first(value, default=None):
    if value is None:
        return default
    return value[0] if isinstance(value, (list, tuple)) and value else default if isinstance(value, (list, tuple)) else value


def _at(values, index, default=None):
    return values[min(index, len(values) - 1)] if values else default


def _inputs(names=None, optional=()):
    names = names or [name for name, _ in REGION_FIELDS]
    return [kind.Input(name, optional=name in optional, **({"force_input": True} if kind != io.Mask else {}))
            for name, kind in REGION_FIELDS if name in names]


def _row(columns, index, image_index=0):
    identity = int(_at(columns.get("region_index"), index, index))
    return {"id": f"{image_index}:{identity}", "image_index": image_index, "index": identity,
            "class_label": str(_at(columns.get("labels"), index, "region")),
            "confidence": float(_at(columns.get("confidences"), index, 1.0)),
            "bbox": [float(_at(columns.get(name), index, 0)) for name in BBOX],
            "crop_region": [int(_at(columns.get(name), index, 0)) for name in CROP]}


def _unpack(detections):
    columns = {name: [] for name, _ in REGION_FIELDS}
    for index, detection in enumerate(detections):
        columns["region_index"].append(index)
        columns["labels"].append(str(detection.label))
        columns["confidences"].append(float(np.asarray(detection.confidence).item()))
        numpy_box = isinstance(detection.bbox, np.ndarray) or any(isinstance(value, np.generic) for value in detection.bbox)
        columns["bbox_dtype"].append(str(np.asarray(detection.bbox).dtype) if numpy_box else "python")
        for name, value in zip(BBOX, detection.bbox):
            columns[name].append(float(value))
        for name, value in zip(CROP, detection.crop_region):
            columns[name].append(int(value))
        columns["masks"].append(torch.from_numpy(detection.cropped_mask).unsqueeze(0))
    return columns


def _box(columns, index):
    values = tuple(_at(columns[name], index) for name in BBOX)
    dtype = _at(columns.get("bbox_dtype"), index, "python")
    return values if dtype == "python" else tuple(np.asarray(values, dtype=dtype))


def _combined(image, masks, columns):
    combined = np.zeros(tuple(image.shape[1:3]), dtype=np.uint8)
    for index, mask in enumerate(masks):
        x1, y1, x2, y2 = (int(_at(columns[name], index)) for name in CROP)
        combined[y1:y2, x1:x2] |= (mask[0].cpu().numpy() * 255).astype(np.uint8)
    return torch.from_numpy(combined.astype(np.float32) / 255).unsqueeze(0)


class _Detection(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.node_id, display_name=cls.title, category="Min_node/Detailer",
            description="Independent detector. Masks, labels, confidence and coordinates are separate standard outputs.",
            inputs=[io.Image.Input("image"), io.Custom(cls.detector_type).Input("detector"),
                    io.Float.Input("threshold", default=0.5, min=0, max=1, step=0.01),
                    io.Int.Input("dilation", default=10, min=-512, max=512),
                    io.Float.Input("crop_factor", default=3.0, min=1, max=10, step=0.1),
                    io.Int.Input("drop_size", default=10, min=1, max=nodes.MAX_RESOLUTION),
                    io.Int.Input("image_index", default=0, min=0, optional=True),
                    io.Custom("DETAILER_HOOK").Input("detailer_hook", optional=True)],
            outputs=[io.Int.Output("count"), *(kind.Output(name, is_output_list=True) for name, kind in REGION_FIELDS),
                     io.Image.Output("preview"), io.String.Output("report")], hidden=HIDDEN,
        )

    @classmethod
    def execute(cls, image, detector, threshold, dilation, crop_factor, drop_size, image_index=0, detailer_hook=None):
        if image.shape[0] != 1:
            raise ValueError("Use Image Batch to Image List and Start Loop to detect each image separately.")
        if cls.detector_type == "BBOX_DETECTOR":
            detector.setAux("face")
        _, detections = detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook=detailer_hook)
        if cls.detector_type == "BBOX_DETECTOR":
            detector.setAux(None)
        columns = _unpack(detections)
        rows = [_row(columns, index, image_index) for index in range(len(detections))]
        marked = annotate(image, rows)
        ui, report = preview_ui(cls, "detect", rows, overview=marked, image_index=image_index)
        return io.NodeOutput(len(detections), *(columns[name] for name, _ in REGION_FIELDS), marked, report, ui=ui)


class FaceDetect(_Detection):
    node_id, title, detector_type = "MinFaceDetect", "YOLO / BBox Detect", "BBOX_DETECTOR"


class SegmentationDetect(_Detection):
    node_id, title, detector_type = "MinSegmentationDetect", "YOLO Segmentation Detect", "SEGM_DETECTOR"


class FaceMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinFaceMask", display_name="SAM / Region Masks", category="Min_node/Detailer", is_input_list=True,
            description="Explicit boxes/masks from any source. SAM_MODEL connects directly from Impact SAMLoader.",
            inputs=[io.Image.Input("image"), *_inputs(optional=("region_index", "labels", "confidences", "bbox_dtype")),
                    io.Boolean.Input("use_sam", default=False), io.Custom("SAM_MODEL").Input("sam_model", optional=True, lazy=True),
                    io.Combo.Input("detection_hint", options=["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"]),
                    io.Int.Input("dilation", default=0, min=-512, max=512), io.Float.Input("threshold", default=0.93, min=0, max=1, step=0.01),
                    io.Int.Input("bbox_expansion", default=0, min=0, max=1000), io.Float.Input("mask_hint_threshold", default=0.7, min=0, max=1, step=0.01),
                    io.Combo.Input("mask_hint_use_negative", options=["False", "Small", "Outter"]), io.Int.Input("image_index", default=0, min=0, optional=True)],
            outputs=[io.Mask.Output("masks", is_output_list=True), io.Mask.Output("combined_mask"), io.Image.Output("overlay"), io.String.Output("report")], hidden=HIDDEN,
        )

    @classmethod
    def check_lazy_status(cls, use_sam, sam_model=None, **kwargs):
        return ["sam_model"] if _first(use_sam) and sam_model is not None and _first(sam_model) is None else []

    @classmethod
    def execute(cls, image, masks, use_sam, detection_hint, dilation, threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative,
                sam_model=None, image_index=None, **columns):
        image, image_index = image[0], _first(image_index, 0)
        columns["masks"] = masks
        refined = masks
        if _first(use_sam):
            if _first(sam_model) is None:
                raise ValueError("Connect SAM_MODEL directly from Impact SAMLoader when use_sam is enabled.")
            # Temporary attributes required by the provider API; these never leave this node.
            regions = [SimpleNamespace(bbox=_box(columns, i),
                                       crop_region=tuple(_at(columns[name], i) for name in CROP), cropped_mask=mask[0].cpu().numpy())
                       for i, mask in enumerate(masks)]
            sam_mask = _impact("core").make_sam_mask(
                _first(sam_model), (tuple(image.shape[1:3]), regions), image, _first(detection_hint), _first(dilation),
                _first(threshold), _first(bbox_expansion), _first(mask_hint_threshold), _first(mask_hint_use_negative))
            combined_sam = (_impact("utils").make_2d_mask(sam_mask).cpu().numpy() * 255).astype(np.uint8)
            refined = []
            for i, mask in enumerate(masks):
                x1, y1, x2, y2 = (int(_at(columns[name], i)) for name in CROP)
                bits = np.bitwise_and((mask[0].cpu().numpy() * 255).astype(np.uint8), combined_sam[y1:y2, x1:x2])
                refined.append(torch.from_numpy(bits.astype(np.float32) / 255).unsqueeze(0))
        combined = _combined(image, refined, columns)
        overlay = mask_overlay(image, combined)
        rows = []
        for index, mask in enumerate(refined):
            row = _row(columns, index, image_index)
            x1, y1, x2, y2 = row["crop_region"]
            rows.append({**row, "mask": mask, "region_preview": image[:, y1:y2, x1:x2], "mask_kind": "SAM refined" if _first(use_sam) else "detector mask"})
        ui, report = preview_ui(cls, "mask", rows, overview=combined, overlay=overlay, image_index=image_index)
        return io.NodeOutput(refined, combined, overlay, report, ui=ui)


class WildcardPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinDetailerWildcardPlan", display_name="Detailer Wildcard / Region Order", category="Min_node/Detailer", is_input_list=True,
                         inputs=[*_inputs(["region_index", "labels", "bbox_dtype", *BBOX, "masks"], optional=("region_index", "labels", "bbox_dtype")),
                                 io.String.Input("wildcard", default="", multiline=True, dynamic_prompts=False),
                                 io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                                 io.Int.Input("image_index", default=0, min=0, optional=True)],
                         outputs=[io.Int.Output("order", is_output_list=True), io.Int.Output("seeds", is_output_list=True),
                                  io.String.Output("texts", is_output_list=True), io.Int.Output("enabled", is_output_list=True),
                                  io.Int.Output("concat"), io.Int.Output("iterations"), io.Int.Output("count"), io.String.Output("report")], hidden=HIDDEN)

    @classmethod
    def execute(cls, masks, wildcard, seed, image_index=None, **columns):
        text, base, image_index = _first(wildcard), _first(seed), _first(image_index, 0)
        concat = text.startswith("[CONCAT]")
        if concat:
            text = text[8:]
        mode, chooser = _impact("wildcards").process_wildcard_for_segs(text)
        order = list(range(len(masks)))
        if mode in ("ASC", "DSC"):
            order.sort(key=lambda i: _box(columns, i)[:2], reverse=mode == "DSC")
        elif mode in ("ASC-SIZE", "DSC-SIZE"):
            order.sort(key=lambda i: (_box(columns, i)[2] - _box(columns, i)[0]) * (_box(columns, i)[3] - _box(columns, i)[1]), reverse=mode == "DSC-SIZE")
        seeds, texts, enabled, rows = [], [], [], []
        stopped = False
        for position, index in enumerate(order):
            label = _at(columns.get("labels"), index, "region")
            active = not stopped and bool(torch.any(masks[index] != 0))
            chosen_seed, chosen_text = None, ""
            if active:
                item = chooser.get(SimpleNamespace(label=label))
                chosen_seed, chosen_text = (None, item) if mode == "LAB" else item
                if chosen_text and chosen_text.strip() == "[STOP]":
                    stopped = True
                active = not stopped and (not chosen_text or chosen_text.strip() != "[SKIP]")
            seeds.append(base + image_index + position if chosen_seed is None else chosen_seed)
            texts.append(chosen_text or "")
            enabled.append(int(active))
            rows.append({"id": f"{image_index}:{index}", "image_index": image_index, "index": index, "class_label": str(label),
                         "confidence": 1.0, "bbox": [float(_at(columns[name], index)) for name in BBOX], "crop_region": [],
                         "processing_order": position, "seed": seeds[-1], "text": texts[-1], "enabled": bool(active)})
        ui, report = preview_ui(cls, "wildcard", rows)
        return io.NodeOutput(order, seeds, texts, enabled, int(concat), max(1, len(order)), len(order), report, ui=ui)


class RegionSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinFaceSelect", display_name="Select Region / Explicit Coordinates", category="Min_node/Detailer", is_input_list=True,
                         inputs=[io.Image.Input("image"), *_inputs(optional=("region_index", "labels", "confidences", "bbox_dtype")),
                                 io.Int.Input("process_index", default=0, min=0), io.Int.Input("order", optional=True, force_input=True),
                                 io.Int.Input("seeds", optional=True, force_input=True), io.String.Input("texts", optional=True, force_input=True),
                                 io.Int.Input("enabled", optional=True, force_input=True), io.String.Input("selected_indices", default="all"),
                                 io.Boolean.Input("preview_only", default=False), io.Int.Input("image_index", default=0, min=0, optional=True)],
                         outputs=[io.Int.Output("process"), io.Int.Output("region_index"), io.String.Output("label"), io.Float.Output("confidence"), io.String.Output("bbox_dtype"),
                                  *(io.Float.Output(name) for name in BBOX), *(io.Int.Output(name) for name in CROP), io.Mask.Output("mask"),
                                  io.Int.Output("seed"), io.String.Output("wildcard"), io.Int.Output("has_wildcard"),
                                  io.Int.Output("sample_enabled"), io.String.Output("report")], hidden=HIDDEN)

    @classmethod
    def execute(cls, image, masks, process_index, selected_indices, preview_only, image_index=None, order=None, seeds=None, texts=None, enabled=None, **columns):
        image, position, image_index = image[0], _first(process_index), _first(image_index, 0)
        selection = RegionSelection(_first(selected_indices))
        sequence = order if order is not None else list(range(len(masks)))
        index = _at(sequence, position, -1)
        columns["masks"] = masks
        rows = []
        for i, mask in enumerate(masks):
            row = _row(columns, i, image_index)
            x1, y1, x2, y2 = row["crop_region"]
            rows.append({**row, "region_preview": image[:, y1:y2, x1:x2], "selected": selection.includes(image_index, row["index"]), "status": "current" if i == index else "available"})
        ui, report = preview_ui(cls, "select", rows)
        if index < 0 or not masks:
            return io.NodeOutput(0, -1, "", 0.0, "python", *([0.0] * 4), *([0] * 4), torch.zeros(1, 1, 1), 0, "", 0, 0, report, ui=ui)
        row = _row(columns, index, image_index)
        text = _at(texts, position, "")
        active = bool(_at(enabled, position, 1)) and bool(torch.any(masks[index] != 0))
        active = active and selection.includes(image_index, row["index"])
        return io.NodeOutput(int(active), row["index"], row["class_label"], _at(columns.get("confidences"), index, 1.0), _at(columns.get("bbox_dtype"), index, "python"),
                             *(_at(columns[name], index) for name in BBOX), *(_at(columns[name], index) for name in CROP),
                             masks[index], _at(seeds, position, image_index + position), text, int(bool(text)),
                             int(active and not _first(preview_only)), report, ui=ui)


class FaceCropUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinFaceCropUpscale", display_name="Crop / Upscale / Feather Masks", category="Min_node/Detailer",
                         inputs=[io.Image.Input("image"), io.Mask.Input("mask"),
                                 *(io.Float.Input(name, force_input=True) for name in BBOX), *(io.Int.Input(name, force_input=True) for name in CROP),
                                 io.Float.Input("guide_size", default=512, min=64, max=nodes.MAX_RESOLUTION, step=8),
                                 io.Boolean.Input("guide_size_for", default=True, label_on="bbox", label_off="crop_region"),
                                 io.Float.Input("max_size", default=1024, min=64, max=nodes.MAX_RESOLUTION, step=8),
                                 io.Int.Input("feather", default=5, min=0, max=100), io.Int.Input("noise_mask_feather", default=20, min=0, max=100),
                                 io.Boolean.Input("noise_mask", default=True), io.Boolean.Input("force_inpaint", default=True),
                                 io.Model.Input("model", optional=True), io.String.Input("bbox_dtype", default="python", optional=True), io.Int.Input("region_index", default=0, optional=True),
                                 io.Int.Input("image_index", default=0, optional=True), io.String.Input("label", default="region", optional=True)],
                         outputs=[io.Image.Output("upscaled"), io.Image.Output("cropped"), io.Mask.Output("blend_mask"), io.Mask.Output("noise_mask"),
                                  io.Int.Output("should_process"), io.Int.Output("crop_width"), io.Int.Output("crop_height"), io.String.Output("report")], hidden=HIDDEN)

    @classmethod
    def execute(cls, image, mask, guide_size, guide_size_for, max_size, feather, noise_mask_feather, noise_mask, force_inpaint,
                model=None, bbox_dtype="python", region_index=0, image_index=0, label="region", **coordinates):
        utils = _impact("utils")
        crop, bbox = tuple(coordinates[name] for name in CROP), tuple(coordinates[name] for name in BBOX)
        if bbox_dtype != "python":
            bbox = tuple(np.asarray(bbox, dtype=bbox_dtype))
        cropped = utils.to_tensor(utils.crop_ndarray4(image.cpu().numpy(), crop))
        raw = mask[0].cpu().numpy()
        blend = utils.tensor_gaussian_blur_mask(utils.to_tensor(raw), feather).squeeze(3)
        noise = utils.tensor_gaussian_blur_mask(raw, noise_mask_feather).squeeze(3) if noise_mask else mask
        h, w = cropped.shape[1:3]
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        active = force_inpaint or not (bh >= guide_size and bw >= guide_size)
        scale = guide_size / min(bw, bh) if guide_size_for else guide_size / min(w, h)
        target_w, target_h = int(w * scale), int(h * scale)
        if model is not None and "aitemplate_keep_loaded" in model.model_options:
            max_size = min(4096, max_size)
        if target_w > max_size or target_h > max_size:
            scale *= max_size / max(target_w, target_h)
            target_w, target_h = int(w * scale), int(h * scale)
        if not force_inpaint and (scale <= 1 or target_w == 0 or target_h == 0):
            active = False
        elif force_inpaint and (scale <= 1 or target_w == 0 or target_h == 0):
            target_w, target_h = w, h
        upscaled = utils.tensor_resize(cropped, target_w, target_h) if active else cropped
        row = {"id": f"{image_index}:{region_index}", "image_index": image_index, "index": region_index, "class_label": label,
               "confidence": 1.0, "bbox": [float(value) for value in bbox], "bbox_dtype": bbox_dtype, "crop_region": list(crop), "cropped": cropped, "upscaled": upscaled,
               "crop_size": [w, h], "upscaled_size": list(upscaled.shape[1:3][::-1]), "status": "ready" if active else "skipped"}
        ui, report = preview_ui(cls, "crop", [row])
        return io.NodeOutput(upscaled, cropped, blend, noise, int(active), w, h, report, ui=ui)


class CropConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinCropConditioning", display_name="Crop Conditioning Masks", category="Min_node/Detailer",
                         inputs=[io.Conditioning.Input("conditioning"), io.Image.Input("image"), *(io.Int.Input(name, force_input=True) for name in CROP)],
                         outputs=[io.Conditioning.Output("conditioning")])

    @classmethod
    def execute(cls, conditioning, image, **coordinates):
        if isinstance(conditioning, str):
            return io.NodeOutput(conditioning)
        crop = tuple(coordinates[name] for name in CROP)
        result = [[condition, {key: _impact("core").crop_condition_mask(value, image, crop) if key == "mask" else value
                              for key, value in details.items()}] for condition, details in conditioning]
        return io.NodeOutput(result)


class WildcardEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinDetailerWildcardEncode", display_name="Detailer Wildcard / LoRA Encode", category="Min_node/Detailer",
                         inputs=[io.Model.Input("model"), io.Clip.Input("clip"), io.String.Input("wildcard", default="", multiline=True, dynamic_prompts=False)],
                         outputs=[io.Model.Output("model"), io.Clip.Output("clip"), io.Conditioning.Output("conditioning"),
                                  io.Conditioning.Output("replacement_conditioning"), io.String.Output("expanded_text")])

    @classmethod
    def execute(cls, model, clip, wildcard):
        if not wildcard:
            return io.NodeOutput(model, clip, [], [], "")
        processed = []
        model, clip, positive = _impact("wildcards").process_with_loras(wildcard, model, clip, processed=processed)
        replacement = [positive[0].copy()]
        if "pooled_output" in positive[0][1]:
            replacement[0][1]["pooled_output"] = positive[0][1]["pooled_output"]
        elif "pooled_output" in replacement[0][1]:
            del replacement[0][1]["pooled_output"]
        return io.NodeOutput(model, clip, positive, replacement, processed[0] if processed else wildcard)


class DetailerModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinDetailerModel", display_name="Detailer Differential Diffusion", category="Min_node/Detailer",
                         inputs=[io.Model.Input("model"), io.Int.Input("noise_mask_feather", default=20, min=0, max=100)], outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, model, noise_mask_feather):
        if noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
            model = _impact("utils").apply_differential_diffusion(model)
        return io.NodeOutput(model)


class DetailerNoiseMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinDetailerNoiseMask", display_name="Attach Detailer Noise Mask", category="Min_node/Detailer",
                         description="Adds the visible mask to a standard LATENT, preserving Impact Pack's N,H,W mask layout.",
                         inputs=[io.Latent.Input("samples"), io.Mask.Input("mask"), io.Boolean.Input("enabled", default=True)], outputs=[io.Latent.Output("samples")])

    @classmethod
    def execute(cls, samples, mask, enabled):
        result = samples.copy()
        if enabled:
            result["noise_mask"] = mask
        return io.NodeOutput(result)


class DetailerSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinDetailerSchedule", display_name="Detailer Sigma Schedule (No Sampling)", category="Min_node/Detailer",
                         inputs=[io.Model.Input("model"), io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                                 io.Combo.Input("scheduler", options=list(comfy.samplers.SCHEDULER_HANDLERS) + ["AYS SDXL", "AYS SD1", "AYS SVD", "GITS[coeff=1.2]", "LTXV[default]", "OSS FLUX", "OSS Wan", "OSS Chroma"]),
                                 io.Int.Input("steps", default=20, min=1, max=10000), io.Float.Input("denoise", default=0.5, min=0.0001, max=1, step=0.01)],
                         outputs=[io.Sigmas.Output("sigmas"), io.Sampler.Output("sampler"), io.Int.Output("total_steps"), io.Int.Output("start_step")])

    @classmethod
    def execute(cls, model, sampler_name, scheduler, steps, denoise):
        total = math.floor(steps / denoise)
        start = total - steps
        sampling = _impact("impact_sampling")
        full = sampling.calculate_sigmas(model, sampler_name, scheduler, total)
        return io.NodeOutput(full[start:start + steps + 1] * 1.0, sampling.ksampler(sampler_name, full), total, start)


class FaceComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinFaceComposite", display_name="Resize Region / Composite", category="Min_node/Detailer",
                         inputs=[io.Image.Input("image"), io.Image.Input("region"), io.Mask.Input("blend_mask"),
                                 *(io.Int.Input(name, force_input=True) for name in CROP),
                                 io.Int.Input("region_index", default=0, optional=True), io.Int.Input("image_index", default=0, optional=True),
                                 io.String.Input("label", default="region", optional=True)],
                         outputs=[io.Image.Output("image"), io.Image.Output("resized_region"), io.Image.Output("alpha_region"), io.String.Output("report")], hidden=HIDDEN)

    @classmethod
    def execute(cls, image, region, blend_mask, region_index=0, image_index=0, label="region", **coordinates):
        utils = _impact("utils")
        x1, y1, x2, y2 = (coordinates[name] for name in CROP)
        if len(region.shape) == 5:
            region = region.squeeze(0)
        resized = utils.tensor_resize(region, x2 - x1, y2 - y1).cpu()
        result = image.clone().cpu()
        mask = blend_mask.unsqueeze(-1)
        utils.tensor_paste(result, resized, (x1, y1), mask)
        alpha = utils.tensor_convert_rgba(resized)
        mask = utils.tensor_resize(mask, *utils.tensor_get_size(resized))
        utils.tensor_putalpha(alpha, mask)
        row = {"id": f"{image_index}:{region_index}", "image_index": image_index, "index": region_index, "class_label": label,
               "confidence": 1.0, "bbox": [], "crop_region": [x1, y1, x2, y2], "refined": resized, "final_refined": resized, "status": "composited"}
        ui, report = preview_ui(cls, "composite", [row], overview=result, image_index=image_index)
        return io.NodeOutput(result, resized, alpha, report, ui=ui)


class DetailerFinalize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinDetailerFinalize", display_name="Detailer Final RGB", category="Min_node/Detailer",
                         inputs=[io.Image.Input("image"), io.Int.Input("region_count", default=1, min=0)], outputs=[io.Image.Output("image")])

    @classmethod
    def execute(cls, image, region_count):
        return io.NodeOutput(_impact("utils").tensor_convert_rgb(image) if region_count else image)


DETAILER_NODES = [FaceDetect, SegmentationDetect, FaceMask, WildcardPlan, RegionSelect, FaceCropUpscale,
                  CropConditioning, WildcardEncode, DetailerModel, DetailerNoiseMask, DetailerSchedule, FaceComposite, DetailerFinalize]
