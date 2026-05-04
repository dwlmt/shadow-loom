# =====================================================================
# Shadow Loom — developer Makefile
# =====================================================================
# Run `make help` for the list of targets. Everything assumes you are
# inside an activated virtualenv (see `make venv`) or that `python` is
# the interpreter you intend to use.
# =====================================================================

PYTHON  ?= python
PIP     ?= $(PYTHON) -m pip
VENV    ?= .venv
ACTIVATE = . $(VENV)/bin/activate
OLLAMA_MODEL ?= qwen3.6:35b

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "; printf "\nUsage: make <target>\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────

.PHONY: setup
setup:  ## One-shot bootstrap (venv + deps + .env). For fresh clones.
	@bash scripts/setup.sh

.PHONY: venv
venv:  ## Create a local virtualenv at .venv (Python 3.13+).
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@echo "Activate with:  source $(VENV)/bin/activate"

.PHONY: install
install:  ## Install the package in editable mode.
	$(PIP) install --upgrade pip
	$(PIP) install -e .

.PHONY: install-dev
install-dev: install  ## Install plus dev tools (pytest, ruff, mypy).
	$(PIP) install pytest pytest-asyncio ruff mypy

.PHONY: env
env:  ## Copy .env.example to .env if .env is missing.
	@if [ -f .env ]; then \
	    echo ".env already exists — leaving it alone."; \
	else \
	    cp .env.example .env && \
	    echo "Wrote .env from .env.example. Edit it to set keys."; \
	fi

# ── Run ───────────────────────────────────────────────────────────

.PHONY: ui
ui:  ## Start the NiceGUI workspace (http://localhost:7860 by default).
	$(PYTHON) -m shadow_loom_ui

.PHONY: mcp
mcp:  ## Start the FastMCP server on stdio.
	$(PYTHON) -m shadow_loom_mcp

.PHONY: pipeline
pipeline:  ## Run the bundled end-to-end pipeline demo.
	$(PYTHON) run_pipeline.py

# ── Ollama ────────────────────────────────────────────────────────

.PHONY: ollama-check
ollama-check:  ## Verify Ollama is reachable and the default model is pulled.
	@command -v ollama >/dev/null 2>&1 || { \
	    echo "ollama not found. Install from https://ollama.com/download"; exit 1; }
	@ollama list 2>/dev/null | grep -q "$(OLLAMA_MODEL)" || { \
	    echo "Model $(OLLAMA_MODEL) not pulled. Run: make ollama-pull"; exit 1; }
	@echo "Ollama OK ($(OLLAMA_MODEL) available)."

.PHONY: ollama-pull
ollama-pull:  ## Pull the default Ollama model (override with OLLAMA_MODEL=...).
	ollama pull $(OLLAMA_MODEL)

# ── Test / lint ───────────────────────────────────────────────────

.PHONY: test
test:  ## Run the fast test suite (no live LLM calls).
	$(PYTHON) -m pytest tests/ --ignore=tests/test_live_e2e.py -q

.PHONY: test-fast
test-fast:  ## Same as `test` but stop on first failure.
	$(PYTHON) -m pytest tests/ --ignore=tests/test_live_e2e.py -q -x

.PHONY: test-live
test-live:  ## Run the live-LLM end-to-end tests (slow; needs Ollama).
	$(PYTHON) -m pytest tests/test_live_e2e.py -q

.PHONY: test-coverage
test-coverage:  ## Run tests with coverage report.
	$(PYTHON) -m pytest tests/ --ignore=tests/test_live_e2e.py \
	    --cov=shadow_loom --cov=shadow_loom_ui --cov=shadow_loom_mcp \
	    --cov-report=term-missing -q

.PHONY: lint
lint:  ## Lint with ruff.
	$(PYTHON) -m ruff check shadow_loom shadow_loom_ui shadow_loom_mcp tests

.PHONY: format
format:  ## Auto-format with ruff.
	$(PYTHON) -m ruff format shadow_loom shadow_loom_ui shadow_loom_mcp tests

.PHONY: typecheck
typecheck:  ## Type-check with mypy.
	$(PYTHON) -m mypy shadow_loom

# ── Docker ────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build:  ## Build the production Docker image.
	docker build -t shadow-loom .

.PHONY: docker-run
docker-run:  ## Run the production image locally on port 7860.
	docker run --rm -it -p 7860:7860 \
	    -e DATABASE_URL=sqlite:////tmp/shadow_loom.db \
	    -e STORAGE_SECRET=local-dev-secret \
	    shadow-loom

.PHONY: docker-up
docker-up:  ## Start the full local stack (Postgres + app) via docker compose.
	docker compose up --build

.PHONY: docker-down
docker-down:  ## Stop and remove the docker compose stack.
	docker compose down

# ── Housekeeping ──────────────────────────────────────────────────

.PHONY: clean
clean:  ## Remove caches, build artifacts, and SQLite DBs.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f shadow_loom.db

# ── Paper ───────────────────────────────────────────────────────────────

PAPER_DIR  := paper
PAPER_NAME := shadow_loom
ARXIV_TARBALL := $(PAPER_DIR)/$(PAPER_NAME)_arxiv.tar.gz

.PHONY: paper
paper:  ## Build paper/shadow_loom.pdf (pdflatex + bibtex + 2 reruns).
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_NAME)
	cd $(PAPER_DIR) && bibtex $(PAPER_NAME)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_NAME)
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode -halt-on-error $(PAPER_NAME)

.PHONY: paper-clean
paper-clean:  ## Remove paper build artefacts (keeps PDF and tarball).
	cd $(PAPER_DIR) && rm -f *.aux *.log *.out *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz \
	    b1.log b2.log b3.log bb.log build1.log build2.log build3.log build_bib.log missfont.log

.PHONY: arxiv
arxiv: paper  ## Build paper/shadow_loom_arxiv.tar.gz suitable for arXiv submission.
	@rm -f $(ARXIV_TARBALL)
	@tmp=$$(mktemp -d) && pkg=$$tmp/$(PAPER_NAME) && mkdir -p $$pkg && \
	    cp $(PAPER_DIR)/$(PAPER_NAME).tex $$pkg/ && \
	    cp $(PAPER_DIR)/references.bib    $$pkg/ && \
	    cp $(PAPER_DIR)/acl.sty           $$pkg/ && \
	    cp $(PAPER_DIR)/acl_natbib.bst    $$pkg/ && \
	    cp $(PAPER_DIR)/$(PAPER_NAME).bbl $$pkg/ && \
	    tar -C $$tmp -czf $(ARXIV_TARBALL) $(PAPER_NAME) && \
	    rm -rf $$tmp
	@echo "Wrote $(ARXIV_TARBALL):"
	@tar -tzf $(ARXIV_TARBALL)
