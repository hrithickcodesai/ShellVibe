PYTHON_FILES = .

.PHONY: help clean format

help:
	@echo "Available commands:"
	@echo "  make format   - Format code and sort imports (via Ruff)"
	@echo "  make clean    - Remove python cache, ruff cache, and build artifacts"


format:
	@echo "--> Formatting code (Black style)..."
	ruff format $(PYTHON_FILES)
	@echo "--> Sorting imports and fixing lints..."
	ruff check --fix $(PYTHON_FILES)


clean:
	@echo "--> Cleaning cache and artifacts..."
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete