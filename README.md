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

## Face Detailer

Requires **ComfyUI-Impact-Pack**. Find these five nodes under **Min_node / Detailer**
and connect their `state` sockets in order:

```text
1. Face Detect -> 2. Face Mask -> 3. Face Crop / Upscale
              -> 4. Face Regenerate -> 5. Face Composite
```

Set all original FaceDetailer inputs and controls on **1. Face Detect**. The first
six outputs of the final node remain the original outputs, with the same list flags:
`image`, `cropped_refined`, `cropped_enhanced_alpha`, `mask`, `detailer_pipe`,
and `cnet_images`.

The implementation reads the installed FaceDetailer functions and inserts pause
points and optional selection gates between stages. With the default all-region
selection it keeps every original operation statement and verifies
that removing the pause points and delegated calls restores the original syntax
tree. It also checks that the on-disk source matches the loaded function bytecode.
Impact Pack source, functions, and model weights are not patched.

Multiple faces must remain sequential: the next crop is taken from the image
after the previous face has been composited. Stage 5 expands subsequent stage
executions automatically, including the original per-image/per-face seeds,
wildcards, hooks, skips, sorting, batch aggregation, and empty outputs. A hidden
continuation node preserves list outputs through ComfyUI graph expansion.

Each node has a scrollable inspector with full-size image links:

| Stage | Inspector and additional outputs |
| --- | --- |
| 1 | Original image with bbox/crop outlines, class labels, confidence, zero-based detection indices; `detections` IMAGE list and JSON `report` |
| 2 | Combined mask, each region's full-image mask and source-region preview, mask type and detection correspondence; `combined_mask`, `region_masks`, `mask_overlay` and `report` |
| 3 | All actual original crops and upscaled inputs, their sizes, processing order and seed; `cropped` and `report` in addition to the existing `upscaled` socket |
| 4 | Input/result thumbnails, per-region checkboxes, select-all/none and class buttons; `selected_indices`, `preview_only` and `report` |
| 5 | Complete inspection summary and `inspection_report`, appended after the unchanged original six outputs |

Gallery IDs use `image_index:mask_index`, starting at zero. Stage 1 uses detection
indices; stage 2 reports the source detection index when its refined mask can be
matched. Its mask indices are used for selection. The class label is the label
returned by the detector, not a newly inferred class. Individual masks are white
inside the target and black outside; the inspector identifies bbox, SAM or
segmentation masks. The original final mask still includes all detected masks.

For choosing regions before diffusion, queue once with stage 4's `preview_only`
enabled. Then check thumbnails (or a class button), disable `preview_only`, and
queue again. `selected_indices` accepts `all` (default), `none`, `0,2`, `0-3`, or
image-specific IDs such as `0:1,1:0`. Selecting fewer regions does not renumber
the original processing indices or seeds. The UI changes apply to the next queue,
not to an already running prompt. Set the seed control to fixed when reproducing
a run across preview/processing queues.

Run the complete five-stage chain, connected to an output such as Preview Image,
to finish all galleries. Scroll within an inspector and click an image to open
it at full size. The saved workflow retains selection settings and inspection
metadata; preview PNGs use ComfyUI's temporary image directory.

All gallery entries are retained, including skipped/not-reached regions. A
detected-region thumbnail is labeled separately when the original pipeline did
not actually crop that region. Later actual crops still come from the image after
earlier selected regions have been composited. The stage 3/4 tensor sockets expose
the current region; their inspectors accumulate all regions. The original final
lists remain sorted as Impact Pack sorts them; `final_crop_index` and
`final_alpha_index` map each region back to those output lists.
As in the original, the detector supplies initial masks in stage 1; stage 2
performs the optional SAM/segmentation refinement.
Connect each `state` socket once; preview/data sockets may branch. Stages execute
again for each queued run because their suspended execution cannot be cached and
reused. An unsupported upstream code layout fails explicitly.

Equivalence tests use the actual installed Impact Pack orchestration, cropping,
masking, resizing, VAE node calls and compositing, with deterministic detector,
VAE and sampler test doubles. They compare intermediate operation traces and all
six outputs with exact tensor equality, including overlapping faces and batches.
These tests do not establish bitwise equality for every real checkpoint/GPU;
hardware nondeterminism and stateful external hooks remain environment-dependent.

A separate real-checkpoint comparison passed on **RTX 4060 Laptop / PyTorch
2.13.0+cu130 / waiIllustriousSDXL_v170**: two overlapping detection regions,
seed 1234, 8 steps, guide size 128, max size 256. All six final outputs and all
19 recorded intermediate operations matched exactly. The test uses real CLIP,
VAE and diffusion sampling with fixed boxes on a synthetic image.

## Tests

From the Windows portable directory:

```powershell
.\python_embeded\python.exe -s -m unittest discover -s .\ComfyUI\custom_nodes\Min_node\tests -v
```

To repeat the real-checkpoint comparison with an installed model:

```powershell
.\python_embeded\python.exe -s .\ComfyUI\custom_nodes\Min_node\tests\compare_detailer_checkpoint.py --checkpoint waiIllustriousSDXL_v170.safetensors --report parity.json
```

The checkpoint test defaults to CUDA; add `--cpu` for CPU execution. It supports
`--steps`, `--guide-size` and `--max-size`. Run it with enough free memory to load
the selected checkpoint alongside any other active ComfyUI instance.
