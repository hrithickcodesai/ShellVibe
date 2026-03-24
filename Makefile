PYTHON ?= uv run python
PYTHON_FILES = src

.PHONY: help format preprocess-tldr convert-0.5b convert-1.5b convert-3b run-0.5b run-1.5b run-3b

help:
	@echo "Available commands:"
	@echo "  make format                  - Format code and sort imports"
	@echo "  make preprocess-tldr         - Parse TLDR markdown files to CSV"
	@echo ""
	@echo "  make run-0.5b                - Run ShellVibe with 0.5B model"
	@echo "  make run-1.5b                - Run ShellVibe with 1.5B model"
	@echo "  make run-3b                  - Run ShellVibe with 3B model"
	@echo ""
	@echo "  make convert-0.5b            - Convert 0.5B: .pt → HF → GGUF"
	@echo "  make convert-1.5b            - Convert 1.5B: .pt → HF → GGUF"
	@echo "  make convert-3b              - Convert 3B: .pt → HF → GGUF"

# run inference
run-0.5b:
	$(PYTHON) src/scripts/shellcode.py --model_size 0.5b

run-1.5b:
	$(PYTHON) src/scripts/shellcode.py --model_size 1.5b

run-3b:
	$(PYTHON) src/scripts/shellcode.py --model_size 3b

# convert to GGUF for llama.cpp
convert-0.5b:
	@echo "Converting 0.5B model..."
	$(PYTHON) src/scripts/export-hf.py \
		--checkpoint pytorch-models/qwen2.5-0.5b-inst-ckpt/best_edit_distance.pt \
		--output_dir hf-models/qwen2.5-0.5b-inst \
		--model_id Qwen/Qwen2.5-0.5B-Instruct
	$(PYTHON) llama.cpp/convert_hf_to_gguf.py \
		hf-models/qwen2.5-0.5b-inst \
		--outfile gguf-models/qwen2.5-0.5b-inst-q8_0.gguf \
		--outtype q8_0

convert-1.5b:
	@echo "Converting 1.5B model..."
	$(PYTHON) src/scripts/export-hf.py \
		--checkpoint pytorch-models/qwen2.5-1.5b-inst-ckpt/best_edit_distance.pt \
		--output_dir hf-models/qwen2.5-1.5b-inst \
		--model_id Qwen/Qwen2.5-1.5B-Instruct
	$(PYTHON) llama.cpp/convert_hf_to_gguf.py \
		hf-models/qwen2.5-1.5b-inst \
		--outfile gguf-models/qwen2.5-1.5b-inst-q8_0.gguf \
		--outtype q8_0

convert-3b:
	@echo "Converting 3B model..."
	$(PYTHON) src/scripts/export-hf.py \
		--checkpoint pytorch-models/qwen2.5-3b-inst-ckpt/best_edit_distance.pt \
		--output_dir hf-models/qwen2.5-3b-inst \
		--model_id Qwen/Qwen2.5-3B-Instruct
	$(PYTHON) llama.cpp/convert_hf_to_gguf.py \
		hf-models/qwen2.5-3b-inst \
		--outfile gguf-models/qwen2.5-3b-inst-q8_0.gguf \
		--outtype q8_0

format:
	@echo "--> Formatting code (Black style)..."
	ruff format $(PYTHON_FILES)
	@echo "--> Sorting imports and fixing lints..."
	ruff check --fix $(PYTHON_FILES)

preprocess-tldr:
	@echo "--> Preprocessing TLDR data..."
	$(PYTHON) -m src.data_processing.tldr_preprocess

