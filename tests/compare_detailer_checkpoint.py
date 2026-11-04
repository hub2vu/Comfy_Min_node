"""Real-checkpoint comparison of the visible graph against original FaceDetailer."""

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
parser.add_argument("--sampler", choices=["KSampler", "SamplerCustom"], default="SamplerCustom")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--guide-size", type=int, default=128)
parser.add_argument("--max-size", type=int, default=256)
parser.add_argument("--report", type=Path)
parser.add_argument("--allow-difference", action="store_true")
parser.add_argument("--generate-example", type=Path)
options = parser.parse_args()
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "custom_nodes/comfyui-impact-pack/modules")]
from comfy.cli_args import args
args.cpu = options.cpu

import execution
import nodes
import numpy as np
from PIL import Image
import torch
from aiohttp import web
from comfy_api.latest import io
from comfy_extras import nodes_custom_sampler, nodes_loop, nodes_math, nodes_primitive
from server import PromptServer

PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None,
    client_id=None, send_sync=lambda *values, **kwargs: None, send_progress_text=lambda *values, **kwargs: None)
import impact.impact_server
from impact import core, impact_pack, utils, util_nodes
from custom_nodes.Min_node.detailer import DETAILER_NODES
from custom_nodes.Min_node.detailer_workflow import build_detailer_prompt
from custom_nodes.Min_node.logic import LOGIC_NODES


class Detector:
    def setAux(self, value):
        pass

    def detect(self, image, *values, **kwargs):
        regions = []
        for crop, bbox in [((2, 2, 22, 24), (6.25, 6.125, 14.005, 14.01)), ((10, 8, 34, 30), (14, 12, 26, 22))]:
            mask = np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.float32)
            mask[2:-2, 2:-2] = 1
            regions.append(core.SEG(None, mask, 0.9, crop, np.asarray(bbox, dtype=np.float32), "face", None))
        return tuple(image.shape[1:3]), regions


detector = Detector()
captured = []


class Inputs(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinExplicitParityInputs", outputs=[io.Image.Output(), io.Custom("BBOX_DETECTOR").Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(torch.linspace(0, 1, 32 * 40 * 3).reshape(1, 32, 40, 3), detector)


class Capture(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MinExplicitParityCapture", is_output_node=True, inputs=[io.Image.Input("image")])

    @classmethod
    def execute(cls, image):
        captured.append(image)
        return io.NodeOutput()


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {name: snapshot(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(snapshot(item) for item in value)
    return value if value is None or isinstance(value, (str, int, float, bool, np.generic)) else type(value).__name__


def exact(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return first.dtype == second.dtype and np.array_equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(exact(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(exact(a, b) for a, b in zip(first, second))
    return first == second


async def main():
    nodes.NODE_CLASS_MAPPINGS.update({node.GET_SCHEMA().node_id: node for node in [*DETAILER_NODES, *LOGIC_NODES,
        nodes_math.MathExpressionNode, nodes_primitive.Int, nodes_primitive.Float, nodes_primitive.Boolean, nodes_custom_sampler.SamplerCustom]})
    nodes.NODE_CLASS_MAPPINGS.update(nodes_loop.NODE_CLASS_MAPPINGS)
    nodes.NODE_CLASS_MAPPINGS.update(FaceDetailer=impact_pack.FaceDetailer, MinExplicitParityInputs=Inputs, MinExplicitParityCapture=Capture,
        ImpactImageBatchToImageList=util_nodes.ImageBatchToImageList, ImageListToImageBatch=util_nodes.ImageListToImageBatch)
    server = SimpleNamespace(client_id=None, send_sync=lambda *values, **kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})

    async def run(graph, name):
        valid, _, outputs, errors = await execution.validate_prompt(name, graph, None)
        if not valid:
            raise AssertionError(errors)
        await executor.execute_async(graph, name, execute_outputs=outputs)
        if not executor.success:
            raise AssertionError(executor.status_messages)

    common = {
        "inputs": {"class_type": "MinExplicitParityInputs", "inputs": {}},
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": options.checkpoint}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["loader", 1], "text": "portrait of a face"}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["loader", 1], "text": "blurry"}},
    }
    if options.generate_example:
        graph = {key: {"class_type": value["class_type"], "inputs": value["inputs"].copy()} for key, value in common.items() if key != "inputs"}
        graph["positive"]["inputs"]["text"] = "portrait of an adult woman, fully clothed, simple light blue shirt, looking at camera, face clearly visible, neutral expression, plain background, high quality"
        graph["negative"]["inputs"]["text"] = "text, logo, watermark, blurry, distorted face"
        graph.update(empty={"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
                     sample={"class_type": "KSampler", "inputs": {"model": ["loader", 0], "seed": 1234, "steps": 20, "cfg": 6.0,
                         "sampler_name": "euler", "scheduler": "normal", "positive": ["positive", 0], "negative": ["negative", 0], "latent_image": ["empty", 0], "denoise": 1.0}},
                     decode={"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["loader", 2]}},
                     capture={"class_type": "MinExplicitParityCapture", "inputs": {"image": ["decode", 0]}})
        await run(graph, "example")
        Image.fromarray(np.clip(captured[-1][0].cpu().numpy() * 255, 0, 255).astype(np.uint8)).save(options.generate_example)
        print(f"Saved example: {options.generate_example}", flush=True)
        return

    settings = dict(seed=1234, guide_size=options.guide_size, max_size=options.max_size, steps=options.steps, cfg=3.0, denoise=0.5, noise_mask_feather=20)
    original = dict(image=["inputs", 0], model=["loader", 0], clip=["loader", 1], vae=["loader", 2], positive=["positive", 0], negative=["negative", 0],
        guide_size_for=True, sampler_name="euler", scheduler="simple", feather=5, noise_mask=True, force_inpaint=True,
        bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0, drop_size=10, bbox_detector=["inputs", 1], wildcard="",
        sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7, sam_mask_hint_use_negative="False", cycle=1, **settings)
    actual, output = build_detailer_prompt(["inputs", 0], ["loader", 0], ["loader", 1], ["loader", 2], ["positive", 0], ["negative", 0], ["inputs", 1], sampler=options.sampler, **settings)
    traces = []
    for split in (False, True):
        trace = {"crop": [], "resize": [], "sampling_input": [], "sampling_output": []}
        graph = {key: {"class_type": value["class_type"], "inputs": value["inputs"].copy()} for key, value in common.items()}
        if split:
            graph.update(actual)
            result = output
        else:
            graph["original"] = {"class_type": "FaceDetailer", "inputs": original.copy()}
            result = ["original", 0]
        graph["capture"] = {"class_type": "MinExplicitParityCapture", "inputs": {"image": result}}
        with ExitStack() as stack:
            for attr, kind in (("crop_ndarray4", "crop"), ("tensor_resize", "resize")):
                function = getattr(utils, attr)
                def record(*values, _function=function, _kind=kind, **kwargs):
                    result = _function(*values, **kwargs)
                    trace[_kind].append(snapshot(result))
                    return result
                stack.enter_context(patch.object(utils, attr, record))
            if not split:
                function = core.impact_sampling.ksampler_wrapper
                def record_original(*values, **kwargs):
                    trace["sampling_input"].append(snapshot(values[8]))
                    result = function(*values, **kwargs)
                    trace["sampling_output"].append(snapshot(result))
                    return result
                stack.enter_context(patch.object(core.impact_sampling, "ksampler_wrapper", record_original))
            elif options.sampler == "SamplerCustom":
                function = nodes_custom_sampler.SamplerCustom.execute.__func__
                def record_custom(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
                    trace["sampling_input"].append(snapshot(latent_image))
                    result = function(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
                    trace["sampling_output"].append(snapshot(result[1]))
                    return result
                stack.enter_context(patch.object(nodes_custom_sampler.SamplerCustom, "execute", classmethod(record_custom)))
            else:
                function = nodes.KSampler.sample
                def record_regular(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
                    trace["sampling_input"].append(snapshot(latent_image))
                    result = function(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
                    trace["sampling_output"].append(snapshot(result[0]))
                    return result
                stack.enter_context(patch.object(nodes.KSampler, "sample", record_regular))
            print("Running explicit graph" if split else "Running original", flush=True)
            await run(graph, f"compare-{split}")
        traces.append(trace)
    comparisons = {name + "_exact": exact(traces[0][name], traces[1][name]) for name in traces[0]}
    comparisons["final_image_exact"] = exact(captured[0], captured[1])
    report = {"checkpoint": options.checkpoint, "device": "cpu" if options.cpu else torch.cuda.get_device_name(), "torch": torch.__version__,
              "sampler": options.sampler, **settings, "overlapping_faces": 2, **comparisons,
              "image_sha256": hashlib.sha256(captured[1].cpu().numpy().tobytes()).hexdigest()}
    if options.report:
        options.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not all(comparisons.values()) and not options.allow_difference:
        raise AssertionError("The explicit graph differs from the reference; see the report.")


asyncio.run(main())
