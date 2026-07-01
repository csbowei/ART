"""
Inference script for ART: runs makeup transfer (mt) or makeup removal (demakeup).
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from torch.amp import autocast

from pipeline_flux_kontext_art import FluxKontextPipeline_Art


# Task configuration and default prompts
TASK_PROMPTS: dict[str, str] = {
    "mt": "Apply makeup transfer",
    "demakeup": "Remove makeup of this person",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and validate task-specific requirements."""
    parser = argparse.ArgumentParser(description="ART inference script.")
    
    parser.add_argument("--task", type=str, required=True, choices=["mt", "demakeup"])
    parser.add_argument("--prompt", type=str, default=None, help="Override the default task prompt.")
    parser.add_argument("--input", type=str, required=True, help="Input face image path.")
    parser.add_argument("--ref", type=str, default=None, help="Reference makeup image (required for --task mt).")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--lora_path", type=str, default="", help="LoRA weights path (optional).")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024, 2048])
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/infer")
    parser.add_argument("--save_concat", action="store_true", help="Also save a side-by-side comparison strip.")

    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = TASK_PROMPTS[args.task]

    if args.task == "mt" and args.ref is None:
        parser.error("--task mt requires --ref.")

    return args


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and weights
    print(f"Loading model from {args.model_path} ...")
    pipeline = FluxKontextPipeline_Art.from_pretrained(
        args.model_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    ).to(args.device, dtype=torch.bfloat16)

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path} ...")
        pipeline.load_lora_weights(args.lora_path)

    pipeline.set_progress_bar_config(disable=True)

    # Prepare inputs
    generator = torch.Generator(device=args.device).manual_seed(args.seed) if args.seed is not None else None
    input_image = Image.open(args.input).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS)
    ref_image = Image.open(args.ref).convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS) if args.ref else None

    # Run inference
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_image = pipeline(
                image=input_image,
                cond=ref_image,  # ref_image=None if task is demakeup
                prompt=args.prompt,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]

    # Save results
    input_stem = Path(args.input).stem
    ref_stem = f"_{Path(args.ref).stem}" if args.ref else ""
    out_path = out_dir / f"{input_stem}{ref_stem}.png"
    gen_image.save(out_path)
    print(f"Saved: {out_path}")

    # Generate comparison strip
    if args.save_concat:
        images = [input_image] + ([ref_image] if ref_image else []) + [gen_image]
        w, h = images[0].size
        canvas = Image.new("RGB", (w * len(images), h))
        for i, img in enumerate(images):
            canvas.paste(img, (i * w, 0))
        
        concat_path = out_dir / f"{input_stem}{ref_stem}_concat.png"
        canvas.save(concat_path)
        print(f"Saved concat: {concat_path}")


if __name__ == "__main__":
    main()