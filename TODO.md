# TODO.md — Implementation State Ledger

**Project:** Pixel Art Auto-Captioner  
**Governed by:** `SPEC.md` `§15` | `AGENTS.md` `§4`  
**Rule:** No code for step N unless all preceding steps are `[x]` completed.

---

| Step | Status | File | Test File | GPU | Test Results |
|------|--------|------|-----------|:---:|--------------|
| 1 | `[ ]` Pending | `common/types.py` | `test_types.py` | No | — |
| 2 | `[ ]` Pending | `common/image_utils.py` | `test_image_utils.py` | No | — |
| 3 | `[ ]` Pending | `ingestion/dataloader.py` | `test_dataloader.py` | No | — |
| 4 | `[ ]` Pending | `common/export_utils.py` | `test_export_utils.py` | No | — |
| 5 | `[ ]` Pending | `captioning/base.py` | (via step 6) | No | — |
| 6 | `[ ]` Pending | `captioning/joycaption.py` | `test_model.py` | Yes | — |
| 7 | `[ ]` Pending | `batch/runner.py` | `test_runner.py` | Yes | — |
| 8 | `[ ]` Pending | `scripts/run_caption.py` | Manual | Yes | — |
| 9 | `[ ]` Pending | `configs/example_config.json` | — | No | — |

---

## Completion Log

_Each entry documents: date/time, step completed, test summary, and the next targeted file._

_(No steps completed yet.)_
