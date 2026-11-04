from contextlib import ExitStack
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes/comfyui-impact-pack/modules")]

from comfy.cli_args import args

args.cpu = True

import execution
import nodes
import numpy as np
import torch
from aiohttp import web
from comfy_api.latest import io
from server import PromptServer

_old_server = getattr(PromptServer, "instance", None)
PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None)
import impact.impact_server
from impact import core, hooks, impact_pack, utils, wildcards
PromptServer.instance = _old_server

from custom_nodes.Min_node.detailer import (
    DETAILER_NODES, FaceComposite, FaceCropUpscale, FaceDetect, FaceMask, FaceRegenerate,
)
from custom_nodes.Min_node.detailer_runner import _split_function


def snapshot(value):
    if isinstance(value, FakeModel):
        return {"model_options": snapshot(value.model_options)}
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(snapshot(item) for item in value)
    return value


class FakeVAE:
    def __init__(self, trace):
        self.trace = trace

    def encode(self, pixels):
        self.trace.append(("encode", snapshot(pixels)))
        return pixels.permute(0, 3, 1, 2).clone()

    def encode_tiled(self, pixels, **kwargs):
        self.trace.append(("encode_tiled", kwargs))
        return self.encode(pixels)

    def decode(self, samples):
        self.trace.append(("decode", snapshot(samples)))
        return samples.permute(0, 2, 3, 1).clamp(0, 1)

    def decode_tiled(self, samples, **kwargs):
        self.trace.append(("decode_tiled", kwargs))
        return self.decode(samples)

    def temporal_compression_decode(self):
        return None

    def spacial_compression_decode(self):
        return 1


class FakeModel:
    def __init__(self):
        self.model_options = {}


class Detector:
    def __init__(self, segments, trace):
        self.segments = segments
        self.trace = trace
        self.override_bbox_by_segm = False

    def setAux(self, value):
        self.trace.append(("setAux", value))

    def detect(self, image, *args, **kwargs):
        self.trace.append(("detect", snapshot(image), args, kwargs))
        return (tuple(image.shape[1:3]), self.segments)


class TraceHook(hooks.DetailerHook):
    def __init__(self, trace, skip=False):
        self.trace, self.skip = trace, skip

    def post_upscale(self, pixels, mask=None):
        self.trace.append(("post_upscale", snapshot(pixels), snapshot(mask)))
        return pixels * 0.97

    def post_encode(self, latent):
        self.trace.append(("post_encode", snapshot(latent)))
        return latent

    def pre_ksample(self, *values):
        self.trace.append(("pre_ksample", snapshot(values)))
        return values

    def post_paste(self, image):
        self.trace.append(("post_paste", snapshot(image)))
        return image + 0.001

    def get_skip_sampling(self):
        self.trace.append(("get_skip_sampling", self.skip))
        return self.skip


class ControlWrapper:
    def __init__(self, trace):
        self.trace = trace

    def apply(self, positive, negative, image, mask):
        self.trace.append(("control", snapshot(image), snapshot(mask)))
        return positive, negative, [image.clone()]

    def doit_ipadapter(self, model):
        self.trace.append(("ipadapter",))
        return model, []


class DetailerTests(unittest.IsolatedAsyncioTestCase):
    def assert_exact(self, first, second):
        self.assertIs(type(first), type(second))
        if isinstance(first, torch.Tensor):
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.device, second.device)
            self.assertTrue(torch.equal(first, second), f"Tensor differs: {first.shape}, {second.shape}")
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_exact(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_exact(a, b)
        else:
            self.assertEqual(first, second)

    def parameters(self, trace):
        image = torch.linspace(0, 1, 32 * 40 * 3).reshape(1, 32, 40, 3)
        segments = []
        for crop, bbox in [((2, 2, 22, 24), (6, 6, 14, 14)), ((10, 8, 34, 30), (14, 12, 26, 22))]:
            mask = np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.float32)
            mask[2:-2, 2:-2] = 1
            segments.append(core.SEG(None, mask, 0.9, crop, bbox, "face", None))
        parameters = {}
        for group in impact_pack.FaceDetailer.INPUT_TYPES().values():
            for name, spec in group.items():
                if len(spec) > 1 and "default" in spec[1]:
                    parameters[name] = spec[1]["default"]
                elif isinstance(spec[0], list):
                    parameters[name] = spec[0][0]
        parameters.update(
            image=image, model=FakeModel(), clip=object(), vae=FakeVAE(trace), seed=1234,
            guide_size=64, max_size=128, steps=2, cfg=3.0,
            positive=[[torch.ones(1, 2, 4), {"mask": torch.ones(1, 32, 40)}]],
            negative=[[torch.zeros(1, 2, 4), {}]], bbox_detector=Detector(segments, trace),
            wildcard="", noise_mask_feather=2,
        )
        return parameters

    def primitives(self, trace):
        stack = ExitStack()

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, *extra, **kwargs):
            trace.append(("sample", seed, steps, cfg, sampler, scheduler, snapshot(positive), snapshot(negative), snapshot(latent), denoise, snapshot(kwargs)))
            noise = torch.rand(latent["samples"].shape, generator=torch.Generator().manual_seed(seed))
            return {**latent, "samples": latent["samples"] * (1 - denoise / 2) + noise * (denoise / 2)}

        def differential(model):
            trace.append(("differential",))
            result = copy.copy(model)
            result.model_options = {**model.model_options, "denoise_mask_function": True}
            return result

        def prompt(text, model, clip):
            trace.append(("wildcard_encode", text))
            return model, clip, [[torch.full((1, 2, 4), len(text) / 100), {}]]

        stack.enter_context(patch.object(core.impact_sampling, "ksampler_wrapper", sample))
        stack.enter_context(patch.object(utils, "apply_differential_diffusion", differential))
        stack.enter_context(patch.object(wildcards, "process_with_loras", prompt))
        for module, name in [
            (utils, "crop_ndarray4"), (utils, "tensor_resize"), (utils, "tensor_gaussian_blur_mask"),
            (utils, "tensor_paste"), (core, "crop_condition_mask"), (core, "segs_to_combined_mask"),
            (utils, "tensor_convert_rgba"), (utils, "tensor_putalpha"), (utils, "empty_pil_tensor"),
        ]:
            original = getattr(module, name)

            def record(*values, _name=name, _function=original, **kwargs):
                trace.append((_name, "in", snapshot(values), snapshot(kwargs)))
                result = _function(*values, **kwargs)
                trace.append((_name, "out", snapshot(result)))
                return result

            stack.enter_context(patch.object(module, name, record))
        return stack

    def run_split(self, parameters):
        state = FaceDetect.execute(**parameters)[0]
        FaceMask.execute(state)
        while True:
            FaceCropUpscale.execute(state)
            FaceRegenerate.execute(state)
            output = FaceComposite.execute(state)
            if state.phase == "complete":
                self.assertIsNone(state.generator)
                return output.result
            self.assertTrue(output.expand)
            if state.phase == "before_detect":
                FaceDetect.execute(continuation=state, **parameters)
                FaceMask.execute(state)

    def compare(self, change=lambda parameters: None):
        trace = []
        parameters = self.parameters(trace)
        change(parameters)
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, {"FaceDetailer": impact_pack.FaceDetailer}), self.primitives(trace):
            random.seed(99)
            expected = impact_pack.FaceDetailer().doit(**parameters)
            expected_trace = snapshot(trace)
            trace.clear()
            random.seed(99)
            actual = self.run_split(parameters)
            self.assert_exact(expected, actual)
            self.assert_exact(expected_trace, snapshot(trace))
        return actual

    def test_exact_pixels_and_operation_trace(self):
        self.compare()

    def test_mismatched_loaded_code_is_rejected(self):
        function = impact_pack.FaceDetailer.doit
        original = function.__code__
        function.__code__ = original.replace(co_consts=(*original.co_consts, "changed runtime"))
        try:
            with self.assertRaisesRegex(RuntimeError, "source and loaded code differ"):
                _split_function(function, "outer")
        finally:
            function.__code__ = original

    def test_batch_overlap_and_cycles(self):
        self.compare(lambda p: p.update(image=p["image"].repeat(2, 1, 1, 1), cycle=2))

    def test_no_faces_empty_masks_dummy_and_skips(self):
        cases = [
            lambda p: p["bbox_detector"].segments.clear(),
            lambda p: [seg.cropped_mask.fill(0) for seg in p["bbox_detector"].segments],
            lambda p: p.update(model="DUMMY"),
            lambda p: p.update(force_inpaint=False, guide_size=4),
            lambda p: p.update(force_inpaint=False, guide_size_for=False, guide_size=16),
            lambda p: p.update(force_inpaint=True, guide_size_for=False, guide_size=16),
            lambda p: p.update(wildcard="[SKIP]"),
            lambda p: p.update(wildcard="[STOP]"),
        ]
        for index, change in enumerate(cases):
            with self.subTest(case=index):
                self.compare(change)

    def test_masks_inpainting_tiling_and_wildcards(self):
        for settings in [
            {"noise_mask": False, "noise_mask_feather": 0, "feather": 0},
            {"inpaint_model": True}, {"tiled_encode": True, "tiled_decode": True},
            {"wildcard": "[ASC]"}, {"wildcard": "[DSC-SIZE]"},
            {"wildcard": "a face"}, {"wildcard": "[CONCAT]a face"},
        ]:
            with self.subTest(settings=settings):
                self.compare(lambda p: p.update(settings))

    def test_hooks_and_control_images(self):
        for skip in (False, True):
            self.compare(lambda p: p.update(detailer_hook=TraceHook(p["vae"].trace, skip)))

        def control(parameters):
            detector = parameters["bbox_detector"]
            detector.segments = [seg._replace(control_net_wrapper=ControlWrapper(detector.trace)) for seg in detector.segments]

        self.compare(control)

    def test_sam_and_segmentation_paths(self):
        def sam(*values):
            return torch.ones(32, 40)

        with patch.object(core, "make_sam_mask", sam):
            self.compare(lambda p: p.update(sam_model_opt=object()))
        for override in (False, True):
            def segmentation(parameters):
                detector = Detector(parameters["bbox_detector"].segments[:1], parameters["vae"].trace)
                detector.override_bbox_by_segm = override
                parameters["segm_detector_opt"] = detector

            self.compare(segmentation)

    def test_schemas_preserve_original_settings_and_outputs(self):
        self.assertEqual(sum(not node.define_schema().is_dev_only for node in DETAILER_NODES), 5)
        actual = FaceDetect.INPUT_TYPES()
        original = impact_pack.FaceDetailer.INPUT_TYPES()
        for group, inputs in original.items():
            for name, expected in inputs.items():
                with self.subTest(input=name):
                    spec = actual[group][name]
                    if isinstance(expected[0], list):
                        self.assertEqual(spec[0], "COMBO")
                        self.assertEqual(spec[1]["options"], expected[0])
                    else:
                        self.assertEqual(spec[0], expected[0])
                    if len(expected) > 1:
                        for key in ("default", "min", "max", "step"):
                            if key in expected[1]:
                                self.assertEqual(spec[1].get(key), expected[1][key])
        self.assertEqual(tuple(FaceComposite.RETURN_TYPES), impact_pack.FaceDetailer.RETURN_TYPES)
        self.assertEqual(tuple(FaceComposite.RETURN_NAMES), impact_pack.FaceDetailer.RETURN_NAMES)
        self.assertEqual(tuple(FaceComposite.OUTPUT_IS_LIST), impact_pack.FaceDetailer.OUTPUT_IS_LIST)


    async def test_expansion_and_repeated_queue(self):
        trace, captured = [], []
        parameters = self.parameters(trace)
        parameters["image"] = parameters["image"].repeat(2, 1, 1, 1)
        object_names = [name for name, value in parameters.items() if not isinstance(value, (str, int, float, bool))]
        original_inputs = impact_pack.FaceDetailer.INPUT_TYPES()
        input_types = {name: spec[0] for group in original_inputs.values() for name, spec in group.items()}

        class Inputs(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id="MinDetailerTestInputs", outputs=[io.Custom(input_types[name]).Output(name) for name in object_names])

            @classmethod
            def execute(cls):
                return io.NodeOutput(*(parameters[name] for name in object_names))

        class Capture(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(
                    node_id="MinDetailerTestCapture", is_output_node=True, is_input_list=True,
                    inputs=[io.AnyType.Input(name) for name in ("image", "crops", "alpha", "mask", "pipe", "control")],
                )

            @classmethod
            def execute(cls, image, crops, alpha, mask, pipe, control):
                captured.append((image[0], crops, alpha, mask[0], pipe[0], control))
                return io.NodeOutput()

        classes = {node.define_schema().node_id: node for node in DETAILER_NODES}
        classes.update(FaceDetailer=impact_pack.FaceDetailer, MinDetailerTestCapture=Capture, MinDetailerTestInputs=Inputs)
        detect_inputs = {**parameters, **{name: ["inputs", index] for index, name in enumerate(object_names)}}
        prompt = {
            "inputs": {"class_type": "MinDetailerTestInputs", "inputs": {}},
            "detect": {"class_type": "MinFaceDetect", "inputs": detect_inputs},
            "mask": {"class_type": "MinFaceMask", "inputs": {"state": ["detect", 0]}},
            "crop": {"class_type": "MinFaceCropUpscale", "inputs": {"state": ["mask", 0]}},
            "regenerate": {"class_type": "MinFaceRegenerate", "inputs": {"state": ["crop", 0]}},
            "composite": {"class_type": "MinFaceComposite", "inputs": {"state": ["regenerate", 0]}},
            "capture": {"class_type": "MinDetailerTestCapture", "inputs": {
                name: ["composite", i] for i, name in enumerate(("image", "crops", "alpha", "mask", "pipe", "control"))
            }},
        }
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, classes), self.primitives(trace):
            server = SimpleNamespace(client_id=None, send_sync=Mock())
            executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
            baseline_prompt = {
                "inputs": prompt["inputs"],
                "original": {"class_type": "FaceDetailer", "inputs": detect_inputs.copy()},
                "capture": {"class_type": "MinDetailerTestCapture", "inputs": {
                    name: ["original", i] for i, name in enumerate(("image", "crops", "alpha", "mask", "pipe", "control"))
                }},
            }
            valid, _, outputs, errors = await execution.validate_prompt("original-detailer", baseline_prompt, None)
            self.assertTrue(valid, errors)
            await executor.execute_async(baseline_prompt, "original-detailer", execute_outputs=outputs)
            self.assertTrue(executor.success, executor.status_messages)
            expected = captured.pop()
            expected_trace = snapshot(trace)
            for index in range(2):
                trace.clear()
                queued_prompt = {key: {"class_type": node["class_type"], "inputs": node["inputs"].copy()} for key, node in prompt.items()}
                valid, _, outputs, errors = await execution.validate_prompt(f"detailer-{index}", queued_prompt, None)
                self.assertTrue(valid, errors)
                await executor.execute_async(queued_prompt, f"detailer-{index}", execute_outputs=outputs)
                self.assertTrue(executor.success, executor.status_messages)
                self.assert_exact(expected, captured[-1])
                self.assert_exact(expected_trace, snapshot(trace))
        self.assertEqual(len(captured), 2)


if __name__ == "__main__":
    unittest.main()
