from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image, ImageOps
from torch.amp import autocast

from pipeline_flux_kontext_art import FluxKontextPipeline_Art


ROOT = Path(__file__).resolve().parent
BASE_MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"
CHECKPOINT_DIR = ROOT / "checkpoints"
PROMPT = "Apply makeup transfer"
SEED = 42
PANEL_MIN_WIDTH = 260
PANEL_MAX_WIDTH = 340
EXAMPLE_COLUMNS = 3
EXAMPLE_GAP = 8
LOCAL_PROXY_BYPASS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


@dataclass(frozen=True)
class CheckpointConfig:
    filename: str
    guidance_scale: float = 3.5
    steps: int = 28


CHECKPOINTS = {
    512: CheckpointConfig("art_transfer_lora_512.safetensors"),
    1024: CheckpointConfig("art_transfer_lora_1024.safetensors", guidance_scale=3.5, steps=28),
}

DISABLED_CHECKPOINTS = {
    2048: CheckpointConfig("art_transfer_lora_2048.safetensors", guidance_scale=3.5, steps=28),
}

ALL_RESOLUTIONS = sorted([*CHECKPOINTS, *DISABLED_CHECKPOINTS])

SOURCE_EXAMPLES = sorted((ROOT / "examples" / "source").glob("*"))
REFERENCE_EXAMPLES = sorted((ROOT / "examples" / "ref").glob("*"))
EMPTY_RESULT = Image.new("RGB", (512, 512), (39, 39, 42))


def _ensure_local_proxy_bypass() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        entries = [entry.strip() for entry in existing.split(",") if entry.strip()]
        merged = entries + [host for host in LOCAL_PROXY_BYPASS if host not in entries]
        os.environ[key] = ",".join(merged)


def load_pipeline() -> FluxKontextPipeline_Art:
    if not torch.cuda.is_available():
        raise RuntimeError("ART requires an NVIDIA CUDA GPU.")

    pipeline = FluxKontextPipeline_Art.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    for resolution, config in CHECKPOINTS.items():
        checkpoint_path = CHECKPOINT_DIR / config.filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {resolution}: {checkpoint_path}")
        pipeline.load_lora_weights(
            str(checkpoint_path),
            adapter_name=f"art_transfer_{resolution}",
        )

    pipeline.set_adapters(f"art_transfer_{min(CHECKPOINTS)}")
    pipeline.to("cuda", dtype=torch.bfloat16)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


pipe = load_pipeline()


def prepare_image(image: Image.Image, resolution: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image.resize((resolution, resolution), Image.Resampling.LANCZOS)


def transfer_makeup(
    source_image: Image.Image | None,
    reference_image: Image.Image | None,
    resolution: int | str,
) -> Image.Image:
    if source_image is None:
        raise gr.Error("Please upload or select a source image.")
    if reference_image is None:
        raise gr.Error("Please upload or select a reference image.")

    resolution = int(resolution)
    if resolution in DISABLED_CHECKPOINTS:
        raise gr.Error(f"Resolution {resolution} is not available yet.")
    if resolution not in CHECKPOINTS:
        raise gr.Error(f"Resolution {resolution} is not available yet.")

    config = CHECKPOINTS[resolution]
    pipe.set_adapters(f"art_transfer_{resolution}")
    source = prepare_image(source_image, resolution)
    reference = prepare_image(reference_image, resolution)
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.bfloat16):
        return pipe(
            image=source,
            cond=reference,
            prompt=PROMPT,
            guidance_scale=config.guidance_scale,
            height=resolution,
            width=resolution,
            num_inference_steps=config.steps,
            generator=generator,
        ).images[0]


def open_example(paths: list[Path], evt: gr.SelectData) -> str:
    index = evt.index
    if isinstance(index, (tuple, list)):
        index = index[0] * EXAMPLE_COLUMNS + index[1]
    return str(paths[int(index)])


def open_source_example(evt: gr.SelectData) -> str:
    return open_example(SOURCE_EXAMPLES, evt)


def open_reference_example(evt: gr.SelectData) -> str:
    return open_example(REFERENCE_EXAMPLES, evt)


def example_columns(paths: list[Path]) -> int:
    return max(1, min(EXAMPLE_COLUMNS, len(paths)))


def example_rows(paths: list[Path]) -> int:
    columns = example_columns(paths)
    return max(1, (len(paths) + columns - 1) // columns)


CSS = f"""
.gradio-container {{
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 28px 28px 32px !important;
}}

.hero {{
    max-width: 1056px;
    margin: 0 auto 30px;
    padding-bottom: 0;
    border-bottom: 0;
    text-align: center;
}}

.hero h1 {{
    max-width: none;
    margin: 0 0 14px;
    font-size: clamp(28px, 3.1vw, 40px);
    line-height: 1.18;
    letter-spacing: 0;
}}

.title-lead {{
    display: block;
    font-size: 1em;
    font-weight: 820;
    color: transparent;
    background: linear-gradient(90deg, #ffffff 0%, #fff1e4 42%, #ff9a3d 100%);
    -webkit-background-clip: text;
    background-clip: text;
}}

.title-tail {{
    display: block;
    margin-top: 4px;
    font-size: 0.64em;
    font-weight: 680;
    color: transparent;
    background: linear-gradient(90deg, #f4f4f5 0%, #d7d2cc 58%, #ffb36a 100%);
    -webkit-background-clip: text;
    background-clip: text;
}}

.hero-note {{
    margin: 12px 0 0;
    color: var(--body-text-color-subdued);
    font-size: 13px;
    line-height: 1.45;
}}

.badge-row {{
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: 7px;
    flex-wrap: wrap;
    margin-bottom: 0;
}}

.badge-row a {{
    display: inline-flex !important;
    align-items: center;
    gap: 7px;
    min-height: 30px;
    padding: 0 11px;
    border: 1px solid var(--border-color-primary);
    border-radius: 999px;
    background: color-mix(in srgb, var(--block-background-fill) 78%, transparent);
    color: var(--body-text-color) !important;
    font-size: 12px;
    font-weight: 650;
    text-decoration: none !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}}

.badge-row a:hover {{
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 10%, var(--block-background-fill));
}}

.badge-icon {{
    display: inline-grid;
    place-items: center;
    width: 17px;
    height: 17px;
    color: var(--color-accent);
}}

.badge-icon svg {{
    width: 15px;
    height: 15px;
    display: block;
    stroke: currentColor;
}}

.badge-text {{
    color: var(--body-text-color);
    font-weight: 720;
}}

.main-row {{
    justify-content: center !important;
    gap: 18px !important;
    align-items: flex-start !important;
}}

.panel-col {{
    flex: 1 1 {PANEL_MIN_WIDTH}px !important;
    max-width: {PANEL_MAX_WIDTH}px !important;
    min-width: {PANEL_MIN_WIDTH}px !important;
    gap: 8px !important;
}}

.panel-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 8px;
    color: var(--body-text-color);
    font-weight: 720;
    font-size: 14px;
    line-height: 1;
}}

.panel-title::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-color-primary);
    opacity: 0.8;
}}

.image-box {{
    width: 100% !important;
    aspect-ratio: 1 / 1 !important;
    margin-top: 0 !important;
    border-radius: 8px !important;
    overflow: visible !important;
}}

.image-box .image-container:not(:fullscreen) .image-frame img {{
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    border-radius: 8px !important;
}}

.example-gallery {{
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    margin-top: 12px !important;
    border-radius: 8px !important;
    overflow: visible !important;
}}

.example-gallery > div,
.example-gallery .grid-wrap,
.example-gallery .grid-container {{
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
    max-height: none !important;
}}

.example-gallery .grid-container {{
    gap: {EXAMPLE_GAP}px !important;
}}

.example-gallery .grid-container,
.example-gallery .grid {{
    grid-template-columns: repeat({EXAMPLE_COLUMNS}, minmax(0, 1fr)) !important;
}}

.example-gallery .thumbnail-item,
.example-gallery .thumbnail-lg,
.example-gallery button {{
    aspect-ratio: 1 / 1 !important;
    height: auto !important;
    min-height: 0 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}}

.example-gallery img {{
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}}

#generate-button {{
    min-height: 52px;
    width: 100% !important;
    margin-top: 10px !important;
    border-radius: 8px !important;
    font-weight: 750 !important;
}}

.control-label {{
    margin: 0 0 9px !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    color: var(--body-text-color);
    font-weight: 760;
    font-size: 14px;
    line-height: 1;
}}

.control-label .prose {{
    padding: 0 !important;
}}

.control-label p {{
    margin: 0 !important;
}}

.resolution-card {{
    margin-top: 12px !important;
    gap: 7px !important;
    padding: 9px 10px 10px !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 0 !important;
}}

.resolution-card > div,
.resolution-card .form {{
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 0 !important;
}}

.resolution-card .panel-title {{
    margin-bottom: 2px !important;
}}

.resolution-radio {{
    width: 100% !important;
    margin-top: -2px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    flex-wrap: nowrap !important;
}}

.resolution-radio .wrap {{
    gap: 7px !important;
}}

.resolution-radio > div,
.resolution-radio fieldset,
.resolution-radio .form,
.resolution-radio .wrap,
.resolution-radio .block {{
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}}

.resolution-radio label {{
    flex: 1 !important;
    text-align: center !important;
    min-height: 38px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}

.resolution-radio label:has(input[value="2048"]),
.resolution-radio label:nth-of-type(3) {{
    opacity: 0.62 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
}}
"""


with gr.Blocks(title="ART Makeup Transfer") as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1><span class="title-lead">Anchoring on Reality:</span><span class="title-tail">Breaking the Pseudo-Target Ceiling in Makeup Transfer</span></h1>
            <div class="badge-row">
                <a href="https://arxiv.org/abs/2606.31089" target="_blank">
                    <span class="badge-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"/>
                            <path d="M14 2v5h5"/>
                            <path d="M9 13h6"/>
                            <path d="M9 17h4"/>
                        </svg>
                    </span>
                    <span class="badge-text">arXiv</span>
                </a>
                <a href="https://csbowei.github.io/ART/" target="_blank">
                    <span class="badge-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <circle cx="12" cy="12" r="10"/>
                            <path d="M2 12h20"/>
                            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                        </svg>
                    </span>
                    <span class="badge-text">Project</span>
                </a>
                <a href="https://github.com/csbowei/ART" target="_blank">
                    <span class="badge-icon">
                        <svg viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">
                            <path d="M12 .5A11.5 11.5 0 0 0 8.36 22.9c.58.11.79-.25.79-.56v-2.14c-3.22.7-3.9-1.38-3.9-1.38-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.78 1.2 1.78 1.2 1.04 1.77 2.72 1.26 3.38.96.11-.75.41-1.26.74-1.55-2.57-.29-5.27-1.28-5.27-5.72 0-1.26.45-2.3 1.2-3.11-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.18 1.19A10.9 10.9 0 0 1 12 5.92c.98 0 1.96.13 2.88.39 2.2-1.5 3.17-1.19 3.17-1.19.63 1.6.23 2.78.11 3.07.75.81 1.2 1.85 1.2 3.11 0 4.45-2.71 5.42-5.29 5.71.42.36.79 1.07.79 2.16v3.17c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/>
                        </svg>
                    </span>
                    <span class="badge-text">GitHub</span>
                </a>
                <a href="https://drive.google.com/drive/folders/1UfUUAk86fWWzYsZcGOzpTmAWa32IwK21?usp=drive_link" target="_blank">
                    <span class="badge-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                            <ellipse cx="12" cy="5" rx="8" ry="3"/>
                            <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/>
                            <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>
                        </svg>
                    </span>
                    <span class="badge-text">MF2K Dataset</span>
                </a>
            </div>
            <p class="hero-note">Transfer reference makeup while preserving the source identity.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False, elem_classes="main-row"):
        with gr.Column(scale=1, min_width=PANEL_MIN_WIDTH, elem_classes=["panel-col", "source-panel"]):
            gr.HTML('<div class="panel-title">Source</div>')
            source = gr.Image(
                value=None,
                type="pil",
                format="png",
                sources=["upload"],
                buttons=["download", "fullscreen"],
                show_label=False,
                elem_classes="image-box",
            )
            if SOURCE_EXAMPLES:
                source_gallery = gr.Gallery(
                    value=[str(path) for path in SOURCE_EXAMPLES],
                    label="Source examples",
                    columns=example_columns(SOURCE_EXAMPLES),
                    rows=example_rows(SOURCE_EXAMPLES),
                    object_fit="cover",
                    allow_preview=False,
                    buttons=[],
                    elem_classes="example-gallery",
                )
                source_gallery.select(
                    fn=open_source_example,
                    inputs=None,
                    outputs=source,
                )

        with gr.Column(scale=1, min_width=PANEL_MIN_WIDTH, elem_classes=["panel-col", "reference-panel"]):
            gr.HTML('<div class="panel-title">Reference</div>')
            reference = gr.Image(
                value=None,
                type="pil",
                format="png",
                sources=["upload"],
                buttons=["download", "fullscreen"],
                show_label=False,
                elem_classes="image-box",
            )
            if REFERENCE_EXAMPLES:
                reference_gallery = gr.Gallery(
                    value=[str(path) for path in REFERENCE_EXAMPLES],
                    label="Reference examples",
                    columns=example_columns(REFERENCE_EXAMPLES),
                    rows=example_rows(REFERENCE_EXAMPLES),
                    object_fit="cover",
                    allow_preview=False,
                    buttons=[],
                    elem_classes="example-gallery",
                )
                reference_gallery.select(
                    fn=open_reference_example,
                    inputs=None,
                    outputs=reference,
                )

        with gr.Column(scale=1, min_width=PANEL_MIN_WIDTH, elem_classes=["panel-col", "result-panel"]):
            gr.HTML('<div class="panel-title">Result</div>')
            result = gr.Image(
                value=EMPTY_RESULT,
                show_label=False,
                type="pil",
                format="png",
                buttons=["download", "fullscreen"],
                interactive=False,
                elem_classes="image-box",
            )
            with gr.Column(elem_classes="resolution-card"):
                gr.HTML('<div class="panel-title">Resolution</div>')
                resolution = gr.Radio(
                    choices=[str(r) for r in ALL_RESOLUTIONS],
                    value=str(min(CHECKPOINTS)),
                    label=None,
                    show_label=False,
                    container=False,
                    elem_classes="resolution-radio",
                )
            generate = gr.Button(
                "Generate",
                variant="primary",
                elem_id="generate-button",
            )

    generate.click(
        fn=transfer_makeup,
        inputs=[source, reference, resolution],
        outputs=result,
        api_name="transfer",
        concurrency_limit=1,
    )


if __name__ == "__main__":
    _ensure_local_proxy_bypass()
    host = os.getenv("ART_SERVER_NAME", "127.0.0.1")
    port = int(os.getenv("ART_SERVER_PORT", "7860"))
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        css=CSS,
    )
