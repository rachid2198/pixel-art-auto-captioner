#!/usr/bin/env python3
"""
pixel-art-auto-captioner

Batch-captions pixel art images using Joy Caption Beta One (LLaVA-based VLM)
with 4-bit quantization for 12GB VRAM (RTX 3060).
"""

import argparse
import logging
import sys
from pathlib import Path
from PIL import Image

import torch
from transformers import (
    BitsAndBytesConfig,
    pipeline,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-caption pixel art images using Joy Caption Beta One."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing pixel art images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where caption .txt files will be saved.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./Models",
        help="Path to directory containing the model files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------
def load_pipeline(model_path: str = "./Models"):
    """Load the LLaVA-based JoyCaption pipeline with 4-bit quantisation."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    if Path(model_path).is_dir():
        logger.info("Loading local model from %s ...", model_path)
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            device_map="cuda:0",
            model_kwargs={"quantization_config": quantization_config}
        )
    else:
        logger.info("Loading model from Hugging Face Hub: %s", model_path)
        pipe = pipeline(
            "image-text-to-text",
            model=model_path,
            device_map="cuda:0", 
            model_kwargs={"quantization_config": quantization_config}
        )

    logger.info("Pipeline loaded successfully.")
    return pipe


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------
def generate_caption(pipe, image_path: Path) -> str:
    """
    Use the pipeline with separate image and text arguments.
    Returns the generated caption text.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error("Failed to open image %s: %s", image_path, exc)
        return ""

    try:
        # Pass image and text directly – the pipeline handles the chat template
        result = pipe(
            image=image,
            text="Write a highly detailed, descriptive caption for this pixel art image.",
            max_new_tokens=256,
            generate_kwargs={"do_sample": True},
        )

        # The pipeline may return a list of dicts with 'generated_text'
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"].strip()
            # Sometimes it returns a list of strings
            if isinstance(first, str):
                return first.strip()

        # Fallback: if result is a plain string
        if isinstance(result, str):
            return result.strip()

        # Last resort: convert whatever we got to string
        return str(result).strip()

    except Exception as exc:
        logger.error("Caption generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Input dir : %s", input_dir)
    logger.info("Output dir: %s", output_dir)

    # Discover image files ---------------------------------------------------
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"**/*{ext}"))  # also recurse

    # Deduplicate (globs may overlap if we have both .jpg and .jpeg patterns)
    image_files = sorted(set(image_files))

    if not image_files:
        logger.warning("No supported image files found in %s", input_dir)
        sys.exit(0)

    logger.info("Found %d image(s) to process.", len(image_files))

    # Load model once --------------------------------------------------------
    try:
        vlm_pipeline = load_pipeline(args.model_dir)
    except Exception as exc:
        logger.exception(
            "Failed to load the JoyCaption pipeline. "
            "Check your GPU memory and dependencies: %s",
            exc,
        )
        sys.exit(1)

    # Process each image -----------------------------------------------------
    success_count = 0
    fail_count = 0

    for img_path in image_files:
        caption_path = output_dir / f"{img_path.stem}.txt"

        # Skip if already processed (idempotency)
        if caption_path.exists():
            logger.info("Skipping (already exists): %s", caption_path.name)
            continue

        logger.info("Processing: %s", img_path.name)

        try:
            caption = generate_caption(vlm_pipeline, img_path)
            if caption:  # Only save if we got a valid caption
                caption_path.write_text(caption, encoding="utf-8")
                logger.info("Saved: %s", caption_path.name)
                success_count += 1
            else:
                logger.warning("Empty caption for %s", img_path.name)
                fail_count += 1
        except Exception as exc:
            logger.error("Failed to caption %s: %s", img_path.name, exc)
            fail_count += 1
            continue

    # Summary ----------------------------------------------------------------
    logger.info(
        "Done. %d succeeded, %d failed, %d total.",
        success_count,
        fail_count,
        len(image_files),
    )


if __name__ == "__main__":
    main()
