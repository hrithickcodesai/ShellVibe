PYTHON_FILES = .
CHECKPOINT ?= checkpoints/best_loss.pt

.PHONY: help clean format prepare train inference inference-interactive

help:
	@echo "Available commands:"
	@echo "  make format                          - Format code and sort imports (via Ruff)"
	@echo "  make clean                           - Remove python cache, ruff cache, and build artifacts"
	@echo "  make prepare                         - Tokenize dataset and write .bin files to data/"
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


prepare:
	@echo "--> Tokenizing dataset..."
	python -m src.prepare


train:
	@echo "--> Starting training..."
	python train.py


inference:
	@if [ -z "$(INSTRUCTION)" ]; then \
		echo "Error: INSTRUCTION is required. Usage: make inference INSTRUCTION='list all files'"; \
		exit 1; \
	fi
	python inference.py --checkpoint $(CHECKPOINT) --instruction "$(INSTRUCTION)"


inference-interactive:
	@echo "--> Starting interactive inference (Ctrl+C to quit)..."
	python inference.py --checkpoint $(CHECKPOINT)


clean:
	@echo "--> Cleaning cache and artifacts..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete