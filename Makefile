.PHONY: help install dev build test lint format clean publish-test publish

help:	## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:	## Install dependencies
	uv sync

dev:	## Install in development mode
	uv pip install -e .

build:	## Build the package
	uv build

test:	## Run tests
	@echo "Running basic syntax and import tests..."
	uv run python -c "from z007 import Agent, create_calculator_tool; print('✓ Import successful')"
	@echo "✓ All tests passed!"

examples:	## Run examples (requires AWS credentials)
	@echo "Running examples (requires AWS credentials)..."
	uv run python examples.py

demo:	## Run integration tests (requires AWS credentials)
	@echo "Running integration tests (requires AWS credentials)..."
	uv run python test.py

lint:	## Run linters
	uv run mypy z007/
	uv run ruff check z007/

format:	## Format code
	uv run ruff format z007/

clean:	## Clean build artifacts
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:	## Run the CLI tool
	uv run z007

publish-test:	## Publish to Test PyPI
	uv publish --publish-url https://test.pypi.org/legacy/

publish:	## Publish to PyPI
	uv publish
