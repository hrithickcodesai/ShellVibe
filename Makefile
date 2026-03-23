PYTHON ?= uv run python
PYTHON_FILES = src
CHECKPOINT ?= checkpoints/best_edit_distance.pt
MODEL_ID  ?= hf-models/qwen-3b-inst
CKPT_STEM  = $(basename $(notdir $(CHECKPOINT)))
HF_DIR    ?= hf-models/$(CKPT_STEM)
GGUF_OUT  ?= gguf-models/$(CKPT_STEM).gguf

.PHONY: help format preprocess-tldr export-hf export-gguf convert-all inference-gguf inference inference-interactive

GGUF ?= gguf-models/$(CKPT_STEM).gguf
INSTRUCTION ?=

help:
	@echo "Available commands:"
	@echo "  make format                                      - Format code and sort imports (via Ruff)"
	@echo "  make preprocess-tldr                             - Parse TLDR markdown files to CSV"
	@echo ""
	@echo "  make export-hf   [CHECKPOINT=...] [MODEL_ID=...]  - Export .pt → HuggingFace (hf-models/<name>/)"
	@echo "  make export-gguf [HF_DIR=...]                    - Convert HF model → GGUF (gguf-models/<name>.gguf)"
	@echo "  make convert-all [MODEL_ID=...]                  - Run full pipeline for all checkpoints/*.pt"
	@echo "  make inference-gguf GGUF=<path> [INSTRUCTION=...]  - Run GGUF inference (interactive if no INSTRUCTION)"
	@echo "  make inference [INSTRUCTION=...]                  - Run .pt inference (interactive if no INSTRUCTION)"

inference:
	$(PYTHON) src/scripts/inference.py \
		--checkpoint $(CHECKPOINT) \
		$(if $(INSTRUCTION),--instruction "$(INSTRUCTION)")

inference-interactive:
	$(PYTHON) src/scripts/inference.py --checkpoint $(CHECKPOINT)

inference-gguf:
	$(PYTHON) src/scripts/inference-gguf.py \
		--model_path $(GGUF) \
		$(if $(INSTRUCTION),--instruction "$(INSTRUCTION)")

export-hf:
	$(PYTHON) src/scripts/export-hf.py \
		--checkpoint $(CHECKPOINT) \
		--output_dir $(HF_DIR) \
		--model_id $(MODEL_ID)

export-gguf:
	$(PYTHON) llama.cpp/convert_hf_to_gguf.py \
		$(HF_DIR) \
		--outfile $(GGUF_OUT) \
		--outtype f16

convert-all:
	@for ckpt in checkpoints/*.pt; do \
		stem=$$(basename $$ckpt .pt); \
		$(MAKE) export-hf  CHECKPOINT=$$ckpt HF_DIR=hf-models/$$stem MODEL_ID=$(MODEL_ID); \
		$(MAKE) export-gguf HF_DIR=hf-models/$$stem GGUF_OUT=gguf-models/$$stem.gguf; \
	done

format:
	@echo "--> Formatting code (Black style)..."
	ruff format $(PYTHON_FILES)
	@echo "--> Sorting imports and fixing lints..."
	ruff check --fix $(PYTHON_FILES)

preprocess-tldr:
	@echo "--> Preprocessing TLDR data..."
	$(PYTHON) -m src.data_processing.tldr_preprocess

