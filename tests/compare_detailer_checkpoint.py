"""Compare all six outputs using an installed checkpoint and real CLIP/VAE/sampling.

Uses fixed overlapping detection boxes on a synthetic image; no image/model download.
Run separately from unittest so device selection happens before importing ComfyUI.
"""

import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--guide-size", type=int, default=128)
parser.add_argument("--max-size", type=int, default=256)
parser.add_argument("--report", type=Path)
options = parser.parse_args()
root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root), str(root / "custom_nodes/comfyui-impact-pack/modules")]

from comfy.cli_args import args

args.cpu = options.cpu

import execution
import nodes
import numpy as np
import torch
from aiohttp import web
from comfy_api.latest import io
from server import PromptServer

PromptServer.instance = SimpleNamespace(
    routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
    client_id=None, send_sync=lambda *values, **kwargs: None,
)
import impact.impact_server
from impact import core, impact_pack, utils
from custom_nodes.Min_node.detailer import DETAILER_NODES


class Detector:
    def setAux(self, value):
        pass

    def detect(self, image, *values, **kwargs):
        segments = []
        for crop, bbox in [((2, 2, 22, 24), (6, 6, 14, 14)), ((10, 8, 34, 30), (14, 12, 26, 22))]:
            mask = np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.float32)
            mask[2:-2, 2:-2] = 1
            segments.append(core.SEG(None, mask, 0.9, crop, bbox, "face", None))
        return tuple(image.shape[1:3]), segments


detector = Detector()
captured = []
trace = []


class Inputs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinParityInputs", outputs=[io.Image.Output(), io.Custom("BBOX_DETECTOR").Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(torch.linspace(0, 1, 32 * 40 * 3).reshape(1, 32, 40, 3), detector)


class Capture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinParityCapture", is_output_node=True, is_input_list=True,
            inputs=[io.AnyType.Input(name) for name in ("image", "crops", "alpha", "mask", "pipe", "control")],
        )

    @classmethod
    def execute(cls, image, crops, alpha, mask, pipe, control):
        captured.append((image[0], crops, alpha, mask[0], pipe[0], control))
        return io.NodeOutput()


def assert_exact(first, second, path="outputs"):
    if type(first) is not type(second):
        raise AssertionError(f"{path}: type differs")
    if isinstance(first, torch.Tensor):
        if first.dtype != second.dtype or first.shape != second.shape or not torch.equal(first, second):
            difference = (first.cpu() - second.cpu()).abs().max().item() if first.shape == second.shape else None
            raise AssertionError(f"{path}: tensor differs (max absolute error {difference})")
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in first:
            assert_exact(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError(f"{path}: length differs")
        for index, (a, b) in enumerate(zip(first, second)):
            assert_exact(a, b, f"{path}[{index}]")
    elif first is not second and first != second:
        raise AssertionError(f"{path}: value differs")


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(snapshot(item) for item in value)
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return type(value).__qualname__


def record_operations():
    stack = ExitStack()
    for module, name in [
        (utils, "crop_ndarray4"), (utils, "tensor_resize"), (utils, "tensor_gaussian_blur_mask"),
        (utils, "to_latent_image"), (utils, "tensor_paste"), (core, "crop_condition_mask"),
        (core, "segs_to_combined_mask"), (core.impact_sampling, "ksampler_wrapper"),
    ]:
        original = getattr(module, name)

        def record(*values, _name=name, _function=original, **kwargs):
            trace.append((_name, "in", snapshot(values), snapshot(kwargs)))
            result = _function(*values, **kwargs)
            trace.append((_name, "out", snapshot(result)))
            return result

        stack.enter_context(patch.object(module, name, record))
    return stack


async def main():
    nodes.NODE_CLASS_MAPPINGS.update({node.GET_SCHEMA().node_id: node for node in DETAILER_NODES})
    nodes.NODE_CLASS_MAPPINGS.update(FaceDetailer=impact_pack.FaceDetailer, MinParityInputs=Inputs, MinParityCapture=Capture)
    parameters = {}
    for group in impact_pack.FaceDetailer.INPUT_TYPES().values():
        for name, spec in group.items():
            if len(spec) > 1 and "default" in spec[1]:
                parameters[name] = spec[1]["default"]
            elif isinstance(spec[0], list):
                parameters[name] = spec[0][0]
    parameters.update(
        image=["inputs", 0], bbox_detector=["inputs", 1],
        model=["loader", 0], clip=["loader", 1], vae=["loader", 2],
        positive=["positive", 0], negative=["negative", 0],
        seed=1234, guide_size=options.guide_size, max_size=options.max_size, steps=options.steps, cfg=3.0, wildcard="",
    )
    common = {
        "inputs": {"class_type": "MinParityInputs", "inputs": {}},
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": options.checkpoint}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["loader", 1], "text": "portrait of a face"}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["loader", 1], "text": "blurry"}},
    }
    server = SimpleNamespace(client_id=None, send_sync=lambda *values, **kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
    expected_trace = None
    for split in (False, True):
        trace.clear()
        prompt = {key: {"class_type": value["class_type"], "inputs": value["inputs"].copy()} for key, value in common.items()}
        if split:
            prompt.update(
                detect={"class_type": "MinFaceDetect", "inputs": parameters.copy()},
                mask={"class_type": "MinFaceMask", "inputs": {"state": ["detect", 0]}},
                crop={"class_type": "MinFaceCropUpscale", "inputs": {"state": ["mask", 0]}},
                regenerate={"class_type": "MinFaceRegenerate", "inputs": {"state": ["crop", 0]}},
                composite={"class_type": "MinFaceComposite", "inputs": {"state": ["regenerate", 0]}},
            )
            output_id = "composite"
        else:
            prompt["original"] = {"class_type": "FaceDetailer", "inputs": parameters.copy()}
            output_id = "original"
        prompt["capture"] = {"class_type": "MinParityCapture", "inputs": {
            name: [output_id, index] for index, name in enumerate(("image", "crops", "alpha", "mask", "pipe", "control"))
        }}
        valid, _, outputs, errors = await execution.validate_prompt(f"parity-{split}", prompt, None)
        if not valid:
            raise AssertionError(errors)
        print("Running split" if split else "Running original", flush=True)
        with record_operations():
            await executor.execute_async(prompt, f"parity-{split}", execute_outputs=outputs)
        if not executor.success:
            raise AssertionError(executor.status_messages)
        if split:
            assert_exact(expected_trace, snapshot(trace), "intermediate_operations")
        else:
            expected_trace = snapshot(trace)
    assert_exact(captured[0], captured[1])
    report = {
        "checkpoint": options.checkpoint, "device": "cpu" if options.cpu else torch.cuda.get_device_name(),
        "torch": torch.__version__, "steps": parameters["steps"], "seed": parameters["seed"],
        "guide_size": parameters["guide_size"], "max_size": parameters["max_size"], "overlapping_faces": 2,
        "all_six_outputs_exact": True,
        "intermediate_operations_exact": True, "recorded_operations": len(trace) // 2,
        "image_sha256": hashlib.sha256(captured[0][0].cpu().numpy().tobytes()).hexdigest(),
    }
    if options.report:
        options.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


asyncio.run(main())
