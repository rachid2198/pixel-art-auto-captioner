# pixel-art-auto-captioner

Captions a large dataset of pixel-art video-game images (≥ 280k JPEGs in 110+ source folders) with
**JoyCaption Beta One** (`fancyfeast/llama-joycaption-beta-one-hf-llava`) for research use.

- **Cloud (RunPod):** vLLM backend, bf16, batched, one process per GPU, sharded (`--shard i/N`), crash-safe resume.
- **Local (Windows, RTX 3060):** HF transformers backend, NF4, for smoke tests.
- Output: JSONL (primary), optional `.txt` files, a run manifest and an HTML QA page.

`SPEC.md` is the design (source of truth), `TODO.md` the progress ledger, `AGENTS.md` the rules for coding agents.
`main.py` is the original prototype and is kept unchanged.

Run every command below **from the repo root**: `configs/sources.json` is resolved against the current directory.

---

## 1. Local setup (Windows)

Python ≥ 3.10. The existing `.env` venv already holds torch 2.12.0+cu130, transformers 5.12, accelerate 1.13 and
bitsandbytes 0.49; the pins below match it, so installing the package changes nothing else.

```powershell
cd F:\Research\Code\pixel-art-auto-captioner

# New venv only (skip if .env exists):
python -m venv .env
.\.env\Scripts\python.exe -m pip install --upgrade pip
.\.env\Scripts\python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

# Install the package (editable) with the local HF backend and the test tools:
.\.env\Scripts\python.exe -m pip install -e ".[local,dev]"
```

- torch is not a package dependency: install the CUDA build first (command above; only torch comes from the
  PyTorch index). `torchvision` / `torchaudio` are not needed.
- Extras (`pyproject.toml`): `local` = `requirements-local.txt` (transformers, accelerate, bitsandbytes);
  `dev` = `requirements-dev.txt` (pytest). The core package needs only Pillow (`requirements.txt`).
- CPU-only use (tests, packing, exports): `pip install -e ".[dev]"` is enough.
- The tests import the installed package, so install once before running them.

### Tests

```powershell
.\.env\Scripts\python.exe -m pytest tests -q
```

The CPU suite needs no torch, transformers or vllm. On Windows the bash and POSIX-only tests of
`tests/test_pack_dataset.py` skip. GPU tests (local, need the weights):

```powershell
$env:JOYCAPTION_MODEL_PATH = ".\Models"
.\.env\Scripts\python.exe -m pytest tests/test_model_gpu.py -q -m gpu
```

---

## 2. Configuration

Precedence: CLI flag > config file > built-in defaults (`common/config.py`). All problems are reported together.

| File | Use |
|---|---|
| `configs/example_config.json` | Cloud defaults: vLLM, bf16, `input_dir` `/workspace/pixel-art`, `output_dir` `/workspace/captions` |
| `configs/local_hf_config.json` | RTX 3060: HF backend, NF4, weights in `./Models`, images in `./input`, output in `./output`, small batches |
| `configs/sources.json` | Source folder → category + display name used in the prompts. **Draft, awaiting your approval** (TODO.md step 11 notes) |

Caption types (`long`, `short`, `tags`, `sd_prompt`) are defined in the config; `generation.enabled_caption_types`
(or `--caption_types`) picks the ones to run. Unknown source folders use `prompts.default_category`.

---

## 3. Running the captioner

```text
python scripts/run_caption.py [--config CONFIG] [--input_dir DIR] [--output_dir DIR]
                              [--backend {vllm,hf}] [--model_path PATH] [--revision REV]
                              [--caption_types TYPE [TYPE ...]] [--shard I/N]
                              [--max_images N] [--batch_size N] [--seed N]
                              [--write_txt | --no-write_txt] [--no-resume]
                              [--gpu_hourly_cost USD] [--dry_run]
                              [--log_level {DEBUG,INFO,WARNING,ERROR}]
```

Local smoke test (images in `./input`, weights in `./Models`):

```powershell
.\.env\Scripts\python.exe scripts\run_caption.py --config configs\local_hf_config.json --max_images 20 --dry_run
.\.env\Scripts\python.exe scripts\run_caption.py --config configs\local_hf_config.json --max_images 20
```

- `--dry_run` discovers, shards and resume-filters, then prints the plan as JSON on stdout (images and captions per
  source and caption type) without loading the model or writing anything.
- `--max_images` limits images still pending **after** resume, so repeated runs advance.
- `--gpu_hourly_cost USD` adds $/1k images and the projected cost of the remaining work to the progress log.
- Logs go to stderr (and to `runtime.log_file` if set).

Exit codes: `0` completed (also an empty plan and every dry run) · `1` config error or model-load / run error ·
`2` interrupted (Ctrl-C / SIGTERM, after flushing) · `3` completed but zero captions succeeded while some were planned.

### Outputs and resume

```text
<output_dir>/
├── captions/shard-<iii>-of-<NNN>.jsonl     # one record per (image, caption type): status ok | flagged | failed
├── failures/shard-<iii>-of-<NNN>.jsonl     # undecodable images and systemic errors
├── txt/<rel_dir>/<stem>.<caption_type>.txt # only with --write_txt (or scripts/export_txt.py)
├── runs/<run_id>.json                      # manifest: config, model, versions, git commit, summary
└── qa/<run_id>.html                        # QA spot-check page (output.qa_sample images; 0 disables)
```

- Resume reads **every** `captions/*.jsonl` in the output dir: an (image, caption type) with an `ok` or `flagged`
  record is skipped; `failed` records are retried on the next run. Re-running the same command continues a killed run.
- Shards are decided by a hash of the relative path, so `--shard i/N` processes are disjoint and can run on different
  machines; merge them by copying their `captions/*.jsonl` into one folder.
- `--no-resume` (or a new output dir) regenerates everything. Captions are never edited; quality problems are flags.

### Other scripts

| Script | Purpose |
|---|---|
| `python scripts/export_txt.py --output_dir DIR [--caption_types ...] [--include_flagged]` | Rebuild `txt/` from the JSONL (latest record per image and type) |
| `python scripts/make_qa_page.py --output_dir DIR --input_dir DIR [--n 200] [--seed 0] [--status ok,flagged] [--source ...]` | Self-contained HTML spot-check page in `qa/` |
| `python scripts/pack_dataset.py --input_dir DIR --out_dir DIR [--target_size_gb 1.5] [--extensions .jpg ...] [--include_sources ...] [--exclude_sources ...]` | Pack the JPGs (read-only on the dataset) into ~1.5 GB `pixel-art-NNN.tar` + `manifest.json` + `SHA256SUMS`; re-running continues |
| `bash scripts/unpack_dataset.sh [TAR_DIR] [TARGET_DIR]` | Cloud side: verify sha256 and extract (defaults `/workspace/dataset-tars`, `/workspace/pixel-art`); safe to re-run |

The Python scripts exit `0` done, `1` usage / config / I-O error, `2` interrupted.

---

## 4. Cloud: Docker image

`docker/Dockerfile` builds on the official vLLM image `vllm/vllm-openai:v0.33.0` (CUDA 13.0, torch, transformers,
vllm 0.33.0), copies `src/`, `scripts/`, `configs/` and installs the package with `--no-deps`. The entrypoint is
`scripts/run_caption.py`; the container command is its flags. No weights, dataset or credentials are in the image;
the model downloads once to `HF_HOME=/workspace/hf-cache` on the volume (set `HF_TOKEN` only if ever needed).

```bash
# From the repo root of a clean clone (keeps Models/, input/, output/ out of the build context):
docker build -f docker/Dockerfile -t <registry-user>/pixel-art-captioner:v1.1 .
docker push <registry-user>/pixel-art-captioner:v1.1
```

The image needs a host driver that supports CUDA 13.0 and a GPU with ≥ 24 GB (RTX 4090, L40S, A6000, A100, …).
The image has not been built yet; the first build happens in the pilot (TODO step 15).

---

## 5. RunPod flow

One **network volume**, in a datacenter that supports RunPod's S3-compatible API, mounted at `/workspace` in every pod:

```text
/workspace/
├── hf-cache/        # HF_HOME, model downloaded once
├── dataset-tars/    # uploaded output of pack_dataset.py
├── pixel-art/       # unpacked dataset (input_dir)
└── captions/        # output_dir, survives pod termination
```

1. **Pack locally** (once; the dataset is only read):

   ```powershell
   .\.env\Scripts\python.exe scripts\pack_dataset.py --input_dir F:\Work\datasets\pixel-art --out_dir F:\Work\datasets\pixel-art-tars
   ```

   `out_dir` must be outside the dataset. Re-running continues an interrupted pack and skips complete tars; delete
   stale tars it warns about before uploading.

2. **Upload** the tars, `manifest.json` and `SHA256SUMS` with the AWS CLI against the volume's S3 API. Create an
   S3 API key in the RunPod console (Settings → S3 API keys; the secret is shown once) and enter it with
   `aws configure`. The bucket is the network volume ID; region and endpoint are the volume's datacenter
   (example `EUR-IS-1`):

   ```powershell
   aws s3 cp F:\Work\datasets\pixel-art-tars s3://<VOLUME_ID>/dataset-tars/ --recursive `
       --region EUR-IS-1 --endpoint-url https://s3api-eur-is-1.runpod.io/ --cli-read-timeout 7200
   aws s3 cp scripts\unpack_dataset.sh s3://<VOLUME_ID>/unpack_dataset.sh `
       --region EUR-IS-1 --endpoint-url https://s3api-eur-is-1.runpod.io/
   ```

   Files over 500 MB go up as multipart uploads automatically. Use `cp --recursive` rather than `sync`
   (RunPod documents `sync` issues). Alternative: `runpodctl send` / `receive` through a running pod.

3. **Unpack** on a cheap CPU pod (any image with bash, coreutils and tar) with the volume attached:

   ```bash
   bash /workspace/unpack_dataset.sh /workspace/dataset-tars /workspace/pixel-art
   ```

   It verifies every tar before extracting and can be re-run. If bash reports `$'\r': command not found`, the script
   was uploaded with Windows line endings: run `sed -i 's/\r$//' /workspace/unpack_dataset.sh` and start it again.

4. **Launch** GPU pods from the Docker image, volume at `/workspace`, one GPU and one shard per pod. The container
   command is the CLI flags, e.g. for two pods in the pilot (500 images each, cost logging on):

   ```text
   --config configs/example_config.json --shard 0/2 --max_images 500 --gpu_hourly_cost 0.69
   --config configs/example_config.json --shard 1/2 --max_images 500 --gpu_hourly_cost 0.69
   ```

   Check first with `--dry_run`. The log reports images/h and, with `--gpu_hourly_cost` (the pod's $/h), $/1k images
   and the projected cost of the remaining work, which the pilot must report before the full run (budget $50–$200).
   Several GPUs on one machine: one process per GPU (`CUDA_VISIBLE_DEVICES=k`, `--shard k/N`).

5. **Interrupted** (spot reclaim, crash)? Start a new pod with the same command; resume continues where it stopped.
   Stop or terminate pods when their run has finished (billing continues while a pod runs).

6. **Download** `captions/` (small) when done, e.g.
   `aws s3 cp s3://<VOLUME_ID>/captions/ .\runpod-captions\ --recursive --region EUR-IS-1 --endpoint-url https://s3api-eur-is-1.runpod.io/`,
   then run `export_txt.py` / `make_qa_page.py` where the images are.

---

## 6. Syncing your PC after the overnight run

The v1.1 work is committed to the `v1.1` branch on GitHub (`origin/v1.1`). The working copy on the PC was edited in
place and holds the same content. To make the PC match the branch exactly:

```text
cd F:\Research\Code\pixel-art-auto-captioner
git fetch origin
git status            # expect only files that are also in origin/v1.1, plus untracked HANDOFF.md/OVERNIGHT_RUN.md
git reset --hard origin/v1.1
```

**Warning:** `git reset --hard` discards local uncommitted edits to tracked files (here they are identical to what was
pushed, so nothing is lost; check `git status` first if you changed anything yourself). Files that were removed in git
(e.g. `tests/test_model.py`) disappear from the PC with this command. Untracked and ignored files (`.env/`, `Models/`,
`input/`, `output/`, `HANDOFF.md`, `OVERNIGHT_RUN.md`) are left alone.
