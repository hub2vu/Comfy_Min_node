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

## Tests

From the Windows portable directory:

```powershell
.\python_embeded\python.exe -s -m unittest discover -s .\ComfyUI\custom_nodes\Min_node\tests -v
```
