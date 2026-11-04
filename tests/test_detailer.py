from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "custom_nodes/comfyui-impact-pack/modules")]
from comfy.cli_args import args
args.cpu = True

import execution
import nodes
import numpy as np
import torch
from aiohttp import web
from comfy_api.latest import io
from comfy_extras import nodes_custom_sampler, nodes_loop, nodes_math, nodes_primitive
from server import PromptServer

_old_server = getattr(PromptServer, "instance", None)
PromptServer.instance = SimpleNamespace(routes=web.RouteTableDef(), add_on_prompt_handler=lambda handler: None)
import impact.impact_server
from impact import core, impact_pack, utils, util_nodes, wildcards
PromptServer.instance = _old_server

from custom_nodes.Min_node.detailer import (
    BBOX, CROP, REGION_FIELDS, DETAILER_NODES, FaceDetect, FaceMask, FaceCropUpscale, FaceComposite,
    RegionSelect, WildcardPlan, DetailerSchedule, DetailerNoiseMask, _unpack,
)
from custom_nodes.Min_node.detailer_preview import RegionSelection
from custom_nodes.Min_node.detailer_workflow import build_detailer_prompt
from custom_nodes.Min_node.logic import LOGIC_NODES


class FakeModel:
    def __init__(self):
        self.model_options = {}


class FakeVAE:
    def encode(self, image):
        return image.permute(0, 3, 1, 2).clone()

    def decode(self, samples):
        return samples.permute(0, 2, 3, 1).clamp(0, 1)


class Detector:
    def __init__(self, regions):
        self.regions = regions
        self.aux = None

    def setAux(self, value):
        self.aux = value

    def detect(self, image, *values, **kwargs):
        return tuple(image.shape[1:3]), self.regions


def fixture(batch=1):
    image = torch.linspace(0, 1, 32 * 40 * 3).reshape(1, 32, 40, 3).repeat(batch, 1, 1, 1)
    regions = []
    for crop, bbox, label in [((2, 2, 22, 24), (6, 6, 14, 14), "face"), ((10, 8, 34, 30), (14, 12, 26, 22), "hand")]:
        mask = np.zeros((crop[3] - crop[1], crop[2] - crop[0]), dtype=np.float32)
        mask[2:-2, 2:-2] = 1
        regions.append(core.SEG(None, mask, np.asarray([0.9], dtype=np.float32), crop, np.asarray(bbox, dtype=np.float32), label, None))
    return image, Detector(regions), FakeModel(), object(), FakeVAE(), [[torch.ones(1, 2, 4), {"mask": torch.ones(1, 32, 40)}]], [[torch.zeros(1, 2, 4), {}]]


def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise, *extras, **kwargs):
    noise = torch.rand(latent["samples"].shape, generator=torch.Generator().manual_seed(seed))
    return {**latent, "samples": latent["samples"] * (1 - denoise / 2) + noise * (denoise / 2)}


def schedule(cls, model, sampler_name, scheduler, steps, denoise):
    return io.NodeOutput(torch.linspace(1, 0, steps + 1), SimpleNamespace(name=sampler_name, scheduler=scheduler, denoise=denoise), int(steps / denoise), int(steps / denoise) - steps)


def native_sample(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
    result = sample(model, noise_seed, len(sigmas) - 1, cfg, sampler.name, sampler.scheduler, positive, negative, latent_image, sampler.denoise)
    return io.NodeOutput(result, result)


def differential(model):
    result = copy.copy(model)
    result.model_options = {**model.model_options, "denoise_mask_function": True}
    return result


def wildcard_encode(text, model, clip, processed=None):
    if processed is not None:
        processed.append(text)
    return model, clip, [[torch.full((1, 2, 4), len(text) / 100), {}]]


def primitives():
    stack = ExitStack()
    stack.enter_context(patch.object(core.impact_sampling, "ksampler_wrapper", sample))
    stack.enter_context(patch.object(utils, "apply_differential_diffusion", differential))
    stack.enter_context(patch.object(wildcards, "process_with_loras", wildcard_encode))
    stack.enter_context(patch.object(DetailerSchedule, "execute", classmethod(schedule)))
    stack.enter_context(patch.object(nodes_custom_sampler.SamplerCustom, "execute", classmethod(native_sample)))
    return stack


class DetailerTests(unittest.IsolatedAsyncioTestCase):
    def assert_exact(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertEqual(first.shape, second.shape)
            self.assertEqual(first.dtype, second.dtype)
            self.assertTrue(torch.equal(first, second), "Tensor values differ")
        else:
            self.assertEqual(first, second)

    def mappings(self):
        mapping = {node.GET_SCHEMA().node_id: node for node in [*DETAILER_NODES, *LOGIC_NODES]}
        mapping.update(nodes_loop.NODE_CLASS_MAPPINGS)
        for node in (nodes_math.MathExpressionNode, nodes_primitive.Int, nodes_primitive.Float, nodes_primitive.Boolean, nodes_custom_sampler.SamplerCustom):
            mapping[node.GET_SCHEMA().node_id] = node
        mapping.update(ImpactImageBatchToImageList=util_nodes.ImageBatchToImageList, ImageListToImageBatch=util_nodes.ImageListToImageBatch)
        return mapping

    async def compare_graph(self, batch=1, **settings):
        image, detector, model, clip, vae, positive, negative = fixture(batch)
        if settings.pop("no_faces", False):
            detector.regions.clear()
        if settings.pop("empty_masks", False):
            for region in detector.regions:
                region.cropped_mask.fill(0)
        captured = []

        class Inputs(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id="MinExplicitTestInputs", outputs=[io.Image.Output(), io.Model.Output(), io.Clip.Output(), io.Vae.Output(),
                                io.Conditioning.Output(), io.Conditioning.Output(), io.Custom("BBOX_DETECTOR").Output()])

            @classmethod
            def execute(cls):
                return io.NodeOutput(image, model, clip, vae, positive, negative, detector)

        class Capture(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(node_id="MinExplicitTestCapture", is_output_node=True, inputs=[io.Image.Input("image")])

            @classmethod
            def execute(cls, image):
                captured.append(image)
                return io.NodeOutput()

        values = dict(guide_size=64, max_size=128, seed=1234, steps=2, cfg=3.0, denoise=0.5, noise_mask_feather=2)
        values.update(settings)
        links = [["inputs", index] for index in range(7)]
        graph, output = build_detailer_prompt(links[0], links[1], links[2], links[3], links[4], links[5], links[6], **values)
        graph.update(inputs={"class_type": "MinExplicitTestInputs", "inputs": {}}, capture={"class_type": "MinExplicitTestCapture", "inputs": {"image": output}})
        original = dict(image=image, model=model, clip=clip, vae=vae, guide_size=values["guide_size"], guide_size_for=True, max_size=values["max_size"],
                        seed=1234, steps=2, cfg=3.0, sampler_name="euler", scheduler="simple", positive=positive, negative=negative, denoise=0.5,
                        feather=5, noise_mask=True, force_inpaint=True, bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0,
                        sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7,
                        sam_mask_hint_use_negative="False", drop_size=10, bbox_detector=detector, wildcard="", noise_mask_feather=2)
        for name in ("guide_size_for", "feather", "noise_mask", "force_inpaint", "wildcard", "noise_mask_feather", "inpaint_model"):
            if name in values:
                original[name] = values[name]
        original["cycle"] = values.get("cycles", 1)
        mapping = self.mappings()
        mapping.update(MinExplicitTestInputs=Inputs, MinExplicitTestCapture=Capture)
        server = SimpleNamespace(client_id=None, send_sync=Mock(), send_progress_text=Mock())
        with patch.dict(nodes.NODE_CLASS_MAPPINGS, mapping), patch.object(PromptServer, "instance", server), primitives():
            random.seed(99)
            expected = impact_pack.FaceDetailer().doit(**original)[0]
            expected_rng = random.getstate()
            random.seed(99)
            valid, _, outputs, errors = await execution.validate_prompt("explicit", graph, None)
            self.assertTrue(valid, errors)
            executor = execution.PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
            await executor.execute_async(graph, "explicit", execute_outputs=outputs)
            self.assertTrue(executor.success, executor.status_messages)
            self.assertEqual(len(captured), 1)
            if not values.get("preview_only") and values.get("selected_indices", "all") == "all":
                self.assert_exact(expected, captured[0])
                self.assertEqual(expected_rng, random.getstate())
        return captured[0], graph

    def test_only_explicit_standard_data_ports(self):
        forbidden = ("SEGS", "MIN_FACE_", "DETAILER_PIPE")
        for node in DETAILER_NODES:
            schema = node.GET_SCHEMA()
            for port in (*schema.inputs, *schema.outputs):
                self.assertNotIn(port.id, ("state", "continuation"))
                self.assertFalse(any(token in port.io_type for token in forbidden))
        self.assertNotIn("MinFaceRegenerate", [node.GET_SCHEMA().node_id for node in DETAILER_NODES])

    def test_detector_is_standalone_and_coordinates_are_numeric(self):
        image, detector, *_ = fixture()
        result = FaceDetect.execute(image, detector, 0.5, 10, 3.0, 10)
        columns = dict(zip((name for name, _ in REGION_FIELDS), result.args[1:1 + len(REGION_FIELDS)]))
        self.assertEqual(result[0], 2)
        self.assertEqual(columns["region_index"], [0, 1])
        self.assertEqual(columns["labels"], ["face", "hand"])
        self.assertTrue(all(type(value) is float for value in columns["bbox_x1"]))
        self.assertEqual(columns["bbox_dtype"], ["float32", "float32"])
        self.assertIsNone(detector.aux)

    def test_sam_is_standalone_and_matches_native_mask_math(self):
        image, detector, *_ = fixture()
        columns = _unpack(detector.regions)
        class Sam:
            def prepare_device(self):
                pass

            def release_device(self):
                pass

            def predict(self, pixels, points, labels, box, threshold):
                mask = np.zeros(pixels.shape[:2], dtype=bool)
                mask[5:28, 4:30] = True
                return [mask]
        model = SimpleNamespace(sam_wrapper=Sam())
        whole = core.make_sam_mask(model, (tuple(image.shape[1:3]), detector.regions), image, "center-1", 0, 0.93, 0, 0.7, "False")
        expected = core.segs_bitwise_and_mask((tuple(image.shape[1:3]), detector.regions), whole)
        actual = FaceMask.execute(image=[image], **columns, use_sam=[True], sam_model=[model], detection_hint=["center-1"],
                                  dilation=[0], threshold=[0.93], bbox_expansion=[0], mask_hint_threshold=[0.7], mask_hint_use_negative=["False"])
        for mask, region in zip(actual[0], expected[1]):
            self.assertTrue(torch.equal(mask[0], torch.from_numpy(region.cropped_mask)))
        self.assertTrue(torch.equal(actual[1][0], core.segs_to_combined_mask(expected)))
        self.assertEqual(FaceMask.check_lazy_status([False], sam_model=[None]), [])
        self.assertEqual(FaceMask.check_lazy_status([True], sam_model=[None]), ["sam_model"])
        self.assertEqual(FaceMask.check_lazy_status([True], sam_model=(None,)), ["sam_model"])
        self.assertEqual(FaceMask.check_lazy_status([True], sam_model=None), [])

    async def test_overlapping_faces_native_loop_graph(self):
        await self.compare_graph()

    async def test_batches_and_cycles(self):
        await self.compare_graph(batch=2, cycles=2)

    async def test_no_faces_and_empty_masks(self):
        await self.compare_graph(no_faces=True)
        await self.compare_graph(empty_masks=True)

    async def test_wildcard_modes(self):
        for text in ("[ASC]", "[DSC]", "[DSC-SIZE]", "[SKIP]", "[STOP]", "a face", "[CONCAT]a face", "[LAB][face]a face[hand]a hand"):
            with self.subTest(wildcard=text):
                await self.compare_graph(wildcard=text)

    async def test_mask_and_size_settings(self):
        for settings in ({"noise_mask": False, "noise_mask_feather": 0}, {"inpaint_model": True}, {"force_inpaint": False, "guide_size_for": False, "guide_size": 64}):
            with self.subTest(settings=settings):
                await self.compare_graph(**settings)

    async def test_preview_and_selection(self):
        preview, _ = await self.compare_graph(preview_only=True)
        image, *_ = fixture()
        self.assertTrue(torch.equal(preview, image))
        none, _ = await self.compare_graph(selected_indices="none")
        self.assertTrue(torch.equal(none, image))
        selected, _ = await self.compare_graph(selected_indices="0:1")
        self.assertFalse(torch.equal(selected, image))


if __name__ == "__main__":
    unittest.main()
