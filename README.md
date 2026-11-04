# Comfy Min Node

Custom nodes for ComfyUI using the V3 node API. Copy or clone this repository into
`ComfyUI/custom_nodes/Min_node`, then restart ComfyUI and refresh the browser.

## Logic

Find these nodes under **Min_node / Logic**:

| Node | Inputs | Output |
| --- | --- | --- |
| AND | a, b | 1 when both inputs are true |
| OR | a, b | 1 when either input is true |
| NOT | a | 1 when the input is false |
| XOR | a, b | 1 when exactly one input is true |
| NAND | a, b | 1 unless both inputs are true |
| NOR | a, b | 1 when both inputs are false |
| XNOR | a, b | 1 when both inputs have the same truth value |

Gate inputs are integers: `0` is false and any nonzero integer is true. Every gate
returns an `INT` containing exactly `0` or `1`. Input widgets default to binary
values; convert a widget to an input to connect another gate.

### Multiplexers

- **MUX 2x1**: `select = 0` routes `input0`; `select = 1` routes `input1`.
- **MUX 4x1**: `select = 0..3` routes `input0..input3` respectively.

Both accept any ComfyUI data type, including IMAGE, STRING, CLIP, MODEL, VAE,
LATENT and custom types. The selected object is passed through unchanged.
Inputs can contain different types; choose a type that the downstream node can
consume. Only the selected branch is evaluated. Unselected inputs may be left
disconnected; selecting a disconnected input reports an error.

## Other nodes

- **Min_node / Prompt**: Prompt Variable Dictionary and Prompt Template Renderer.
- **Min_node / Routing**: Broadcast by Input Name, Named Set and Named Get
  (requires `cg-use-everywhere`).

## Face Detailer: explicit data connections

Requires ComfyUI with the native **Start Loop / End Loop** nodes (tested on
0.43.0), Impact Pack and Impact Subpack. No session object, `state`, `SEGS`,
`DETAILER_PIPE`, coroutine runner or JSON data bundle is passed between nodes.
JSON reports are inspection outputs, not computation inputs.

Detection exposes separate standard lists: region index, class label, confidence,
bbox coordinate dtype, four bbox coordinates, four crop coordinates and MASKs.
`bbox_dtype` preserves the provider's float32 arithmetic while the coordinate
ports themselves contain ordinary FLOAT values.

- **YOLO / BBox Detect** accepts IMAGE and BBOX_DETECTOR directly from Impact's
  **UltralyticsDetectorProvider**. No model/CLIP/VAE/wildcard inputs are required.
- **YOLO Segmentation Detect** is the equivalent independent SEGM_DETECTOR input.
- **SAM / Region Masks** accepts IMAGE, explicit boxes/crop coordinates, crop-local
  MASKs and SAM_MODEL directly from **SAMLoader (Impact)**. It produces refined
  crop-local MASKs, a full-image combined mask and an overlay. `use_sam=False`
  preserves detector masks; the connected SAM loader is lazy until enabled.
- **Detailer Wildcard / Region Order** handles ordering, labels, `[CONCAT]`,
  `[SKIP]`, `[STOP]`, per-region text and seeds separately from detection.
- **Select Region / Explicit Coordinates** chooses by `image:index`, exposes the
  selected coordinates/mask/text/seed, and emits separate process/sample flags.
- **Crop / Upscale / Feather Masks** uses only its visible image, mask, numeric
  inputs and settings. Its optional MODEL input only preserves the legacy
  AITemplate size cap when that integration is actually being used.
- **Crop Conditioning Masks** and **Detailer Wildcard / LoRA Encode** expose
  standard CONDITIONING/MODEL/CLIP outputs. Concatenation and switches are visible
  existing nodes in the workflow.
- **Attach Detailer Noise Mask** preserves the original N,H,W noise-mask layout
  in a standard LATENT. **Detailer Differential Diffusion** applies the original
  conditional model patch. Neither performs sampling.
- **Resize Region / Composite** receives IMAGE, MASK and explicit crop coordinates.

Image encoding, decoding, inpaint conditioning and sampling use the **existing
ComfyUI nodes**. There is no Min sampling node. The sigma adapter only constructs
SIGMAS/SAMPLER inputs using Impact's schedule conventions; it never evaluates the
model or runs a sampler.

### Saved workflows

- [Existing KSampler](examples/face_detailer_ksampler.json)
- [Existing SamplerCustom, original denoised output](examples/face_detailer_denoised.json)

Both are ordinary native workflow JSON files with visible image/region/cycle
loops. Parallel coordinate wires are intentional: individual values and masks
can be inspected or replaced. Each region crops the IMAGE carried from the
previous composite, preserving overlap order. Existing VAEEncode, VAEDecode,
InpaintModelConditioning, ConditioningConcat and math/MUX nodes remain visible.

Default controls match ordinary FaceDetailer: all regions selected, preview off,
SAM off, guide 512, max 1024, steps 20, CFG 8, denoise 0.5, blend feather 5,
noise-mask feather 20 and one cycle. Shared settings are standard primitive nodes.
Set `preview_only` on the selection node to inspect crops/upscales while skipping
VAE encoding, diffusion and compositing, then turn it off to process the selection.
A native Loop's normal execution cache should remain enabled.

The KSampler variant implements the explicitly requested existing KSampler node.
It is not bitwise identical to Impact's denoised path: a real GPU comparison found
an approximately 1.9e-6 intermediate-latent difference. The denoised variant uses
ComfyUI's **existing SamplerCustom.denoised_output**. On RTX 4060 Laptop,
PyTorch 2.13.0+cu130 and waiIllustriousSDXL_v170, its crop/resize results, sampler
input/output latents and final image matched the original exactly (two overlapping
regions, 8 steps, guide 128, max 256, seed 1234).

The old automatic session pipeline and its opaque hook/refiner/ControlNet wrapper
propagation have been removed. Advanced hooks, refiners, custom noise and guider
variants must be represented by the corresponding existing nodes and validated
for that graph; the comparison above is not a universal claim for such setups.

On another installation, copy `examples/Min_node_detailer_example.png` to
`ComfyUI/input/` and select your installed checkpoint/YOLO/SAM files in their
loader nodes. API-format companions have `.api.json` names. The example image is
a locally generated, fully clothed adult portrait for testing detection.

### Inspection and selection

Scroll within a node inspector and click images for full-size previews. Indexing
starts at zero; `0:1` means image 0, detection 1. Selection accepts `all`, `none`,
`0,2`, ranges such as `0-3`, or image-specific IDs such as `0:1,1:0`. The selected
index is not renumbered when other regions are excluded. Processing order and seed
are explicit outputs of the wildcard plan. Galleries accumulate native Loop
iterations using UI metadata; no image-processing state is hidden in them.

### Validation

From the portable directory:

```powershell
.\python_embeded\python.exe -s -m unittest discover -s .\ComfyUI\custom_nodes\Min_node\tests -v
```

For a real-checkpoint comparison of the explicit graph with original FaceDetailer:

```powershell
.\python_embeded\python.exe -s .\ComfyUI\custom_nodes\Min_node\tests\compare_detailer_checkpoint.py --checkpoint waiIllustriousSDXL_v170.safetensors --report parity.json
```

Add `--sampler KSampler --allow-difference` to measure the ordinary KSampler path.
`--cpu`, `--steps`, `--guide-size`, and `--max-size` are also supported.
