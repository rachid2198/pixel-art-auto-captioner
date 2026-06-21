# TODO.md — Implementation State Ledger

**Project:** Pixel Art Auto-Captioner  
**Governed by:** `SPEC.md` `§15` | `AGENTS.md` `§4`  
**Rule:** No code for step N unless all preceding steps are `[x]` completed.

---

| Step | Status | File | Test File | GPU | Target Engine | Test Results |
|------|--------|------|-----------|:---:|:---:|--------------|
| 1 | `[x]` Complete | `common/types.py` | `test_types.py` | No | `qwen/qwen3-coder:free` | 10 passed, 0 failed |
| 2 | `[ ]` Pending | `common/image_utils.py` | `test_image_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` | — |
| 3 | `[ ]` Pending | `ingestion/dataloader.py` | `test_dataloader.py` | No | `openrouter/deepseek/deepseek-v4-flash` | — |
| 4 | `[ ]` Pending | `common/export_utils.py` | `test_export_utils.py` | No | `openrouter/deepseek/deepseek-v4-flash` | — |
| 5 | `[ ]` Pending | `captioning/base.py` | (via step 6) | No | `qwen/qwen3-coder:free` | — |
| 6 | `[ ]` Pending | `captioning/joycaption.py` | `test_model.py` | Yes | `openrouter/deepseek/deepseek-v4-pro` | — |
| 7 | `[ ]` Pending | `batch/runner.py` | `test_runner.py` | Yes | `openrouter/deepseek/deepseek-v4-pro` | — |
| 8 | `[ ]` Pending | `scripts/run_caption.py` | Manual | Yes | `openrouter/deepseek/deepseek-v4-flash` | — |
| 9 | `[ ]` Pending | `configs/example_config.json` | — | No | `qwen/qwen3-coder:free` | — |

---

## Completion Log

- **2026-06-16 — Step 1 complete:** `common/types.py` (ImageRecord + CaptionRecord dataclasses). Tests: 10 passed, 0 failed. Cross-platform path fixes applied (WindowsPath handling in asdict). Next targeted file: `common/image_utils.py`.
