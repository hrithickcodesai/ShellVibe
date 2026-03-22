PYTHON ?= uv run python
PYTHON_FILES = src
CHECKPOINT ?= checkpoints/best_edit_distance.pt

.PHONY: help clean format prepare train inference inference-interactive preprocess-tldr generate-data split-data generate-singles-v2 generate-combinations-v2 generate-v2 generate-v2-1b

help:
	@echo "Available commands:"
	@echo "  make format                          - Format code and sort imports (via Ruff)"
	@echo "  make clean                           - Remove python cache, ruff cache, and build artifacts"
	@echo "  make preprocess-tldr                 - Parse TLDR markdown files to CSV"
	@echo "  make generate-data                   - Generate single-command synthetic data using LLM (v1, 6 variations)"
	@echo "  make split-data                      - Split combined data into train/test CSVs"
	@echo "  make prepare                         - Tokenize dataset (bin-packing) and write .bin files"
	@echo "  make train                           - Run SFT training"
	@echo "  make inference INSTRUCTION='...'     - Run single inference with best checkpoint"
	@echo "  make inference-interactive           - Start interactive inference shell"
	@echo ""
	@echo "  make generate-singles-v2             - v2: generate per-tier singles (T1=12, T2=8, T3=6 variations)"
	@echo "  make generate-combinations-v2        - v2: generate combinations (0.5b scale)"
	@echo "  make generate-v2                     - v2 full pipeline for 0.5b target (preprocess → singles → combos → split → prepare)"
	@echo "  make generate-v2-1b                  - v2 full pipeline for 1b target"
	@echo ""
	@echo "  Override checkpoint: make inference CHECKPOINT=checkpoints/last.pt INSTRUCTION='...'"

format:
	@echo "--> Formatting code (Black style)..."
	ruff format $(PYTHON_FILES)
	@echo "--> Sorting imports and fixing lints..."
	ruff check --fix $(PYTHON_FILES)

preprocess-tldr:
	@echo "--> Preprocessing TLDR data..."
	$(PYTHON) -m src.data_processing.tldr_preprocess

