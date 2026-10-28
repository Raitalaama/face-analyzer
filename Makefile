.DEFAULT_GOAL := help
.PHONY: help install format lint test check pre-commit clean models

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install git hooks
	uv sync
	uv run pre-commit install

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

lint: ## Check code without modifying anything
	uv run ruff check .
	uv run ruff format --check .

test: ## Run the test suite
	uv run pytest

check: lint test ## Run all checks (what CI will run)

pre-commit: ## Run all pre-commit hooks on every file
	uv run pre-commit run --all-files

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

models: ## Download and prepare models specified in the manifest file
	uv run python -m face_analyzer.artifacts --manifest-path configs/models.yaml --artifact-dir models
