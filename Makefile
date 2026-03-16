PYTHON ?= uv run python
PYTHON_FILES = src
CHECKPOINT ?= checkpoints/best_edit_distance.pt

.PHONY: help clean format prepare train inference inference-interactive preprocess-tldr generate-data combine-data split-data

help:
	@echo "Available commands:"
	@echo "  make format                          - Format code and sort imports (via Ruff)"
	@echo "  make clean                           - Remove python cache, ruff cache, and build artifacts"
	@echo "  make preprocess-tldr                 - Parse TLDR markdown files to CSV"
	@echo "  make generate-data                   - Generate synthetic data using LLM"
	@echo "  make combine-data                    - Combine TLDR and synthetic data"
	@echo "  make split-data                      - Split combined data into train/test CSVs"
	@echo "  make prepare                         - Tokenize dataset (bin-packing) and write .bin files"
	@echo "  make train                           - Run SFT training"
	@echo "  make inference INSTRUCTION='...'     - Run single inference with best checkpoint"
	@echo "  make inference-interactive           - Start interactive inference shell"
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

generate-data:
	@echo "--> Generating synthetic data..."
	$(PYTHON) -m src.data_processing.generate

combine-data:
	@echo "--> Combining data sources..."
	$(PYTHON) -m src.data_processing.generate_combinations

split-data:
	@echo "--> Splitting data into train/test CSVs..."
	$(PYTHON) -m src.data_processing.split

prepare:
	@echo "--> Tokenizing dataset into .bin files..."
	$(PYTHON) -m src.data_processing.prepare

train:
	@echo "--> Starting training..."
	$(PYTHON) -m src.training.train

inference:
	@if [ -z "$(INSTRUCTION)" ]; then \
		echo "Error: INSTRUCTION is required. Usage: make inference INSTRUCTION='list all files'"; \
		exit 1; \
	fi
	$(PYTHON) -m src.models.inference --checkpoint $(CHECKPOINT) --instruction "$(INSTRUCTION)"

inference-interactive:
	@echo "--> Starting interactive inference (Ctrl+C to quit)..."
	$(PYTHON) -m src.models.inference --checkpoint $(CHECKPOINT)

clean:
	@echo "--> Cleaning cache and artifacts..."
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete