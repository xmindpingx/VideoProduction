"""Make 14 GB VRAM versions of ComfyUI's official "LTX-2.3 image + audio to video" template (video_ltx2_3_ia2v).

    python3 make_14gb_workflows.py path/to/video_ltx2_3_ia2v.json [out_dir]

The template ships with ComfyUI (comfyui-workflow-templates, MIT). In a ComfyUI install it is at
  <python site-packages>/comfyui_workflow_templates_json/templates/video_ltx2_3_ia2v.json
You can also pass your own export of that workflow (Workflow > Export) to keep your prompt and inputs.

Writes:
  ltx23_ia2v_14gb.json       same models as the template (FP8 checkpoint), lower-memory settings
  ltx23_ia2v_14gb_gguf.json  the same, with the 22B model loaded from a GGUF file (needs ComfyUI-GGUF)
"""
import copy
import json
import sys
from pathlib import Path

SUBGRAPH_NAME = "Video Generation (LTX-2.3)"

# Defaults on the "Video Generation (LTX-2.3)" node. Width and height are multiples of 64 because the template
# renders a first pass at half size, and LTX latents are 32 px blocks: anything else is silently rounded down.
# At 24 fps every whole number of seconds gives a valid LTX frame count (8n + 1).
WIDTH, HEIGHT, FPS, SECONDS = 384, 576, 24, 10

# VAE decode in 512 px tiles and 64-frame chunks. The template decodes the whole clip in one temporal chunk
# (temporal_size 4096), which is the largest single memory spike after sampling.
TILED_DECODE = [512, 64, 64, 8]  # tile_size, overlap, temporal_size, temporal_overlap

# Prompt enhance makes the 12B Gemma model write a longer prompt (up to 2048 tokens) before encoding it: an extra
# text-generation pass. Off here; switch it back on in the node if you want it.
PROMPT_ENHANCE = False

GGUF_DEFAULT = "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"


def find_main(workflow):
    sg = next((s for s in workflow.get("definitions", {}).get("subgraphs", []) if s.get("name") == SUBGRAPH_NAME), None)
    if sg is None:
        raise SystemExit(f'No "{SUBGRAPH_NAME}" subgraph found: is this the LTX-2.3 image+audio-to-video template?')
    top = next((n for n in workflow["nodes"] if n.get("type") == sg["id"]), None)
    if top is None:
        raise SystemExit("The workflow does not use its Video Generation subgraph node")
    return sg, top


def node(sg, node_type):
    found = [n for n in sg["nodes"] if n["type"] == node_type]
    if len(found) != 1:
        raise SystemExit(f"Expected exactly one {node_type} node in the subgraph, found {len(found)}")
    return found[0]


WIDGET_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")


def set_param(sg, top, input_name, value):
    """Set a value on the top-level node. Its widgets_values follow the subgraph's widget-type inputs in order
    (prompt, prompt_enhance, width, height, audio_start, duration, fps, seed, ckpt_name, ...)."""
    names = [i["name"] for i in sg["inputs"] if i.get("type") in WIDGET_TYPES]
    if len(names) != len(top["widgets_values"]) or input_name not in names:
        raise SystemExit(f"Unexpected widget layout on the Video Generation node ({input_name})")
    top["widgets_values"][names.index(input_name)] = value


def low_memory(workflow):
    w = copy.deepcopy(workflow)
    sg, top = find_main(w)
    set_param(sg, top, "value_1", WIDTH)
    set_param(sg, top, "value_2", HEIGHT)
    set_param(sg, top, "value_3", FPS)
    set_param(sg, top, "value_4", float(SECONDS))
    set_param(sg, top, "value_5", PROMPT_ENHANCE)
    node(sg, "VAEDecodeTiled")["widgets_values"] = list(TILED_DECODE)
    return w


def with_gguf(workflow):
    """Load the diffusion model from a GGUF file; the FP8 checkpoint is still used for the video VAE, the audio VAE
    and the text-encoder connector weights. The distilled LoRA is bypassed because the GGUF is the distilled model."""
    w = copy.deepcopy(workflow)
    sg, _ = find_main(w)
    ckpt = node(sg, "CheckpointLoaderSimple")
    lora = node(sg, "LoraLoaderModelOnly")
    model_link = next(l for l in sg["links"] if l["origin_id"] == ckpt["id"] and l["origin_slot"] == 0 and l["target_id"] == lora["id"])

    state = sg["state"]
    new_id = max(state["lastNodeId"], max(n["id"] for n in sg["nodes"])) + 1
    state["lastNodeId"] = new_id
    x, y = ckpt["pos"]
    sg["nodes"].append({
        "id": new_id,
        "type": "UnetLoaderGGUF",
        "pos": [x, y - 140],
        "size": [400, 60],
        "flags": {},
        "order": ckpt.get("order", 0),
        "mode": 0,
        "inputs": [],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [model_link["id"]]}],
        "title": "Unet Loader (GGUF): LTX-2.3 22B distilled",
        "properties": {"cnr_id": "ComfyUI-GGUF", "Node name for S&R": "UnetLoaderGGUF"},
        "widgets_values": [GGUF_DEFAULT],
    })
    # rewire: GGUF MODEL -> LoRA node input (instead of the checkpoint's MODEL)
    model_link["origin_id"], model_link["origin_slot"] = new_id, 0
    ckpt_model_out = next(o for o in ckpt["outputs"] if o["name"] == "MODEL")
    ckpt_model_out["links"] = [l for l in (ckpt_model_out.get("links") or []) if l != model_link["id"]]
    lora["mode"] = 4  # bypass: MODEL passes straight through
    return w


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
    workflow = json.loads(src.read_text())
    base = low_memory(workflow)
    (out / "ltx23_ia2v_14gb.json").write_text(json.dumps(base, indent=1))
    (out / "ltx23_ia2v_14gb_gguf.json").write_text(json.dumps(with_gguf(base), indent=1))
    print(f"Wrote {out / 'ltx23_ia2v_14gb.json'} and {out / 'ltx23_ia2v_14gb_gguf.json'}")


if __name__ == "__main__":
    main()
