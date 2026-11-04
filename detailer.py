import comfy.samplers
import nodes
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .detailer_runner import DetailerRun


Detected = io.Custom("MIN_FACE_DETECTED")
Masked = io.Custom("MIN_FACE_MASKED")
Prepared = io.Custom("MIN_FACE_PREPARED")
Regenerated = io.Custom("MIN_FACE_REGENERATED")


def _bind_inspection(cls, state, stage):
    state.inspection.bind(stage, cls.hidden.unique_id if cls.hidden else None)


def _face_inputs():
    return [
        io.Image.Input("image"), io.Model.Input("model"), io.Clip.Input("clip"), io.Vae.Input("vae"),
        io.Float.Input("guide_size", default=512, min=64, max=nodes.MAX_RESOLUTION, step=8),
        io.Boolean.Input("guide_size_for", default=True, label_on="bbox", label_off="crop_region"),
        io.Float.Input("max_size", default=1024, min=64, max=nodes.MAX_RESOLUTION, step=8),
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
        io.Int.Input("steps", default=20, min=1, max=10000),
        io.Float.Input("cfg", default=8.0, min=0.0, max=100.0),
        io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
        io.Combo.Input("scheduler", options=list(comfy.samplers.SCHEDULER_HANDLERS) + [
            "AYS SDXL", "AYS SD1", "AYS SVD", "GITS[coeff=1.2]", "LTXV[default]", "OSS FLUX", "OSS Wan", "OSS Chroma",
        ]),
        io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
        io.Float.Input("denoise", default=0.5, min=0.0001, max=1.0, step=0.01),
        io.Int.Input("feather", default=5, min=0, max=100, step=1),
        io.Boolean.Input("noise_mask", default=True, label_on="enabled", label_off="disabled"),
        io.Boolean.Input("force_inpaint", default=True, label_on="enabled", label_off="disabled"),
        io.Float.Input("bbox_threshold", default=0.5, min=0.0, max=1.0, step=0.01),
        io.Int.Input("bbox_dilation", default=10, min=-512, max=512, step=1),
        io.Float.Input("bbox_crop_factor", default=3.0, min=1.0, max=10.0, step=0.1),
        io.Combo.Input("sam_detection_hint", options=["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"]),
        io.Int.Input("sam_dilation", default=0, min=-512, max=512, step=1),
        io.Float.Input("sam_threshold", default=0.93, min=0.0, max=1.0, step=0.01),
        io.Int.Input("sam_bbox_expansion", default=0, min=0, max=1000, step=1),
        io.Float.Input("sam_mask_hint_threshold", default=0.7, min=0.0, max=1.0, step=0.01),
        io.Combo.Input("sam_mask_hint_use_negative", options=["False", "Small", "Outter"]),
        io.Int.Input("drop_size", default=10, min=1, max=nodes.MAX_RESOLUTION, step=1),
        io.Custom("BBOX_DETECTOR").Input("bbox_detector"),
        io.String.Input("wildcard", default="", multiline=True, dynamic_prompts=False),
        io.Int.Input("cycle", default=1, min=1, max=10, step=1),
        io.Custom("SAM_MODEL").Input("sam_model_opt", optional=True),
        io.Custom("SEGM_DETECTOR").Input("segm_detector_opt", optional=True),
        io.Custom("DETAILER_HOOK").Input("detailer_hook", optional=True),
        io.Boolean.Input("inpaint_model", default=False, optional=True, label_on="enabled", label_off="disabled"),
        io.Int.Input("noise_mask_feather", default=20, min=0, max=100, step=1, optional=True),
        io.Custom("SCHEDULER_FUNC").Input("scheduler_func_opt", optional=True),
        io.Boolean.Input("tiled_encode", default=False, optional=True, label_on="enabled", label_off="disabled"),
        io.Boolean.Input("tiled_decode", default=False, optional=True, label_on="enabled", label_off="disabled"),
    ]


class FaceDetect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinFaceDetect", display_name="1. Face Detect", category="Min_node/Detailer",
            description="Starts the original Impact Pack FaceDetailer with the same settings. Connect the state outputs through all five stages.",
            inputs=[*_face_inputs(), Regenerated.Input("continuation", optional=True, advanced=True)],
            outputs=[Detected.Output("state"), io.Custom("SEGS").Output("segs"),
                     io.Image.Output("detections", is_output_list=True), io.String.Output("report")],
            hidden=[io.Hidden.unique_id], not_idempotent=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # A suspended execution must never be reused by a subsequent queue run.
        return float("nan")

    @classmethod
    def execute(cls, continuation=None, **parameters):
        if continuation is None:
            face_detailer = nodes.NODE_CLASS_MAPPINGS.get("FaceDetailer")
            if face_detailer is None:
                raise RuntimeError("Min_node/Detailer requires ComfyUI-Impact-Pack and its FaceDetailer node.")
            run = DetailerRun(face_detailer, parameters)
            _bind_inspection(cls, run, "detect")
            run.advance()
        else:
            run = continuation
        _bind_inspection(cls, run, "detect")
        run.require("before_detect")
        run.advance()
        run.require("detected")
        return io.NodeOutput(run, run.value, [run.inspection.last_annotated], run.inspection.report("detect"), ui=run.inspection.ui("detect"))


class FaceMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinFaceMask", display_name="2. Face Mask", category="Min_node/Detailer",
            inputs=[Detected.Input("state")],
            outputs=[Masked.Output("state"), io.Custom("SEGS").Output("segs"),
                     io.Mask.Output("combined_mask"), io.Mask.Output("region_masks", is_output_list=True),
                     io.Image.Output("mask_overlay", is_output_list=True), io.String.Output("report")],
            hidden=[io.Hidden.unique_id], not_idempotent=True,
        )

    @classmethod
    def execute(cls, state):
        _bind_inspection(cls, state, "mask")
        state.require("detected")
        state.advance()
        state.require("masked")
        inspection = state.inspection
        return io.NodeOutput(state, state.value, inspection.last_combined_mask, inspection.last_region_masks,
                             [inspection.last_mask_overlay], inspection.report("mask"), ui=inspection.ui("mask"))


class FaceCropUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinFaceCropUpscale", display_name="3. Face Crop / Upscale", category="Min_node/Detailer",
            description="Crops the current composited image and runs the original preprocessing and resize. Faces are processed sequentially.",
            inputs=[Masked.Input("state")],
            outputs=[Prepared.Output("state"), io.Image.Output("upscaled", is_output_list=True),
                     io.Image.Output("cropped", is_output_list=True), io.String.Output("report")],
            hidden=[io.Hidden.unique_id], not_idempotent=True,
        )

    @classmethod
    def execute(cls, state):
        _bind_inspection(cls, state, "crop")
        state.require("masked", "before_crop")
        state.advance()
        while state.phase in ("before_crop", "crop_input", "region_ready"):
            state.advance()
        state.require("cropped", "regenerated", "finish")
        cropped = [state.inspection.cropped] if state.inspection.cropped is not None else []
        return io.NodeOutput(state, [state.value] if state.phase == "cropped" else [], cropped,
                             state.inspection.report("crop"), ui=state.inspection.ui("crop"))


class FaceRegenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinFaceRegenerate", display_name="4. Face Regenerate", category="Min_node/Detailer",
            inputs=[Prepared.Input("state"),
                    io.String.Input("selected_indices", default="all", optional=True,
                                    tooltip="all, none, 0,2, 0-3, or image:index such as 0:1,1:0. Indices refer to stage 2 masks."),
                    io.Boolean.Input("preview_only", default=False, optional=True,
                                     tooltip="Inspect all regions without VAE encoding or diffusion sampling, then select thumbnails and queue again.")],
            outputs=[Regenerated.Output("state"), io.Image.Output("refined", is_output_list=True), io.String.Output("report")],
            hidden=[io.Hidden.unique_id], not_idempotent=True,
        )

    @classmethod
    def execute(cls, state, selected_indices="all", preview_only=False):
        _bind_inspection(cls, state, "regenerate")
        state.regeneration_options = {"selected_indices": selected_indices, "preview_only": preview_only}
        state.inspection.configure(selected_indices, preview_only)
        state.require("cropped", "regenerated", "finish")
        if state.phase == "cropped":
            state.advance(state.inspection.current_selected)
        state.require("regenerated", "finish")
        refined = [state.value] if state.inspection.current_selected and state.phase == "regenerated" and state.value is not None else []
        state.inspection.publish("regenerate")
        return io.NodeOutput(state, refined, state.inspection.report("regenerate"), ui=state.inspection.ui("regenerate"))


class FaceComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinFaceComposite", display_name="5. Face Composite", category="Min_node/Detailer",
            description="Composites in the original order, repeats stages for remaining faces/images, and returns the six original FaceDetailer outputs.",
            inputs=[Regenerated.Input("state")],
            outputs=[
                io.Image.Output("image"), io.Image.Output("cropped_refined", is_output_list=True),
                io.Image.Output("cropped_enhanced_alpha", is_output_list=True), io.Mask.Output("mask"),
                io.Custom("DETAILER_PIPE").Output("detailer_pipe"), io.Image.Output("cnet_images", is_output_list=True),
                io.String.Output("inspection_report"),
            ],
            hidden=[io.Hidden.unique_id], not_idempotent=True, enable_expand=True,
        )

    @classmethod
    def execute(cls, state):
        _bind_inspection(cls, state, "composite")
        state.require("regenerated", "finish")
        state.advance(state.inspection.current_selected if state.phase == "regenerated" else None)
        while state.phase in ("composited", "finish"):
            state.advance()
        state.require("before_crop", "before_detect", "complete")
        if state.phase == "complete":
            return io.NodeOutput(*state.result, state.inspection.report(), ui=state.inspection.ui("composite"))
        graph = GraphBuilder()
        if state.phase == "before_detect":
            detect = graph.node("MinFaceDetect", **state.parameters, continuation=state)
            mask = graph.node("MinFaceMask", state=detect.out(0))
            crop_state = mask.out(0)
        else:
            crop_state = state
        crop = graph.node("MinFaceCropUpscale", state=crop_state)
        regenerate = graph.node("MinFaceRegenerate", state=crop.out(0), **state.regeneration_options)
        composite = graph.node("MinFaceDetailerContinue", state=regenerate.out(0))
        return io.NodeOutput(*(composite.out(i) for i in range(len(cls.RETURN_TYPES))), expand=graph.finalize(), ui=state.inspection.ui("composite"))


class FaceDetailerContinue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = FaceComposite.define_schema()
        schema.node_id = "MinFaceDetailerContinue"
        schema.display_name = "Face Detailer Continue (internal)"
        schema.category = "Min_node/Detailer/internal"
        schema.is_dev_only = True
        # Expansion flattens linked list outputs before applying OUTPUT_IS_LIST.
        # Carry each list as one value until the public Composite node unpacks it.
        schema.outputs = [
            io.Custom("MIN_FACE_IMAGE_LIST").Output(output.id) if output.is_output_list else output
            for output in schema.outputs
        ]
        return schema

    @classmethod
    def execute(cls, state):
        return FaceComposite.execute(state)


DETAILER_NODES = [FaceDetect, FaceMask, FaceCropUpscale, FaceRegenerate, FaceComposite, FaceDetailerContinue]
