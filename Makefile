.PHONY: venv install run run-llm clean reset-history lint format check help

PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
HISTORY := history.jsonl

## —— Setup ——————————————————————————————————————

venv:  ## Create virtual environment
	python3 -m venv .venv

install: venv  ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## —— Run ——————————————————————————————————————

run: ## Run with random oracle (use ARGS="" for extra flags)
	$(PYTHON) main.py $(ARGS)

run-llm: ## Run with LLM oracle (use ARGS="" for extra flags)
	$(PYTHON) main.py --oracle llm $(ARGS)

## —— Cleanup ————————————————————————————————————

clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

reset-history: ## Delete history.jsonl
	rm -f $(HISTORY)

## —— Code quality ———————————————————————————————

lint: ## Run ruff linter
	$(PYTHON) -m ruff check .

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format .

check: lint ## Alias: run all checks

## —— Help ——————————————————————————————————————

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
