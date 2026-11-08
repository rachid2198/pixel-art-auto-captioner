# Cloud image for the vLLM backend (SPEC v1.1 §11.2, §15).
#
# Build from the repo root (a clean clone keeps the build context small):
#   docker build -f docker/Dockerfile -t <registry-user>/pixel-art-captioner:v1.1 .
#
# The base image provides CUDA 13.0, torch, transformers and vllm==0.33.0
# (the version the vLLM backend was written against, TODO step 9). This image
# only adds the package (pillow is already in the base image). No model
# weights, dataset, captions or credentials are baked in: the model downloads
# once into HF_HOME on the mounted volume; dataset and output dirs are mounted.
FROM vllm/vllm-openai:v0.33.0

# Model cache on the RunPod network volume (SPEC §11.4: /workspace/hf-cache).
# Pass HF_TOKEN at run time if ever needed; never bake it in.
ENV HF_HOME=/workspace/hf-cache \
    PYTHONUNBUFFERED=1

# configs/sources.json is resolved against the current directory, so the
# image runs from the repo copy.
WORKDIR /app

# Copy only what the package and scripts need (no Models/, input/, output/,
# .env/, data/ or .git/ from the build context).
COPY pyproject.toml README.md requirements.txt ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# --no-deps: keep the base image's pinned torch/transformers/vllm/pillow
# untouched. PIP_BREAK_SYSTEM_PACKAGES allows the install into the image's
# system Python if it is marked externally managed (Ubuntu 24.04 base).
# The checks fail the build if the base image does not carry vllm 0.33.0 or
# the CLI cannot start.
RUN PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --no-deps --no-cache-dir . \
    && python3 -c "import importlib.metadata as m, sys; v = m.version('vllm'); print('vllm', v); sys.exit(0 if v.split('+')[0] == '0.33.0' else 'expected vllm 0.33.0, got ' + v)" \
    && python3 -c "import PIL, pixel_art_auto_captioner as p; print('pixel-art-auto-captioner', p.__version__, '| pillow', PIL.__version__)" \
    && python3 scripts/run_caption.py --help > /dev/null

# Replace the base image's API-server entrypoint with the captioning CLI.
# Pass the CLI flags as the container command, e.g.
#   docker run --gpus all -v /workspace:/workspace <image> \
#       --config configs/example_config.json --shard 0/1 --gpu_hourly_cost 0.69
ENTRYPOINT ["python3", "/app/scripts/run_caption.py"]
CMD ["--help"]
