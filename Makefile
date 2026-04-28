# LangGraph Recruiter Agent — convenience targets.
#
# Tested with GNU make on macOS/Linux. Windows users without `make` can run
# the same commands directly (each target is a single shell line); see the
# README "Quickstart" section for the equivalent commands.

PYTHON ?= python
PIP    ?= pip

# Default target prints the help so `make` with no args is informative.
.DEFAULT_GOAL := help

.PHONY: help install verify run cli smoke test calibrate calibrate-write \
        ablation ab-compare seed-labels clean-cache check

help: ## Show this help.
	@echo "LangGraph Recruiter Agent — make targets"
	@echo ""
	@echo "  Setup"
	@echo "    install         Install Python dependencies"
	@echo "    verify          Check MySQL connectivity (.env must be populated)"
	@echo "    seed-labels     Load 21 synthetic labels for the calibration demo"
	@echo ""
	@echo "  Run"
	@echo "    run             Launch the Streamlit UI on http://localhost:8501"
	@echo "                    (sidebar nav: 'app' for search, '2_Label_queue' for labeling)"
	@echo "    cli             One-off CLI search (see SEARCH_QUERY env var)"
	@echo ""
	@echo "  Test"
	@echo "    test            Run pytest unit + snapshot tests"
	@echo "    smoke           Headless end-to-end test across 10 prompts"
	@echo "    check           install + verify + test + smoke (full sanity sweep)"
	@echo ""
	@echo "  Calibration / evaluation"
	@echo "    calibrate       Dry-run the calibrator (no weights written)"
	@echo "    calibrate-write Fit and write config/weights.json + history snapshot"
	@echo "    ablation        5-row Jaccard@3 ablation table over held-out prompts"
	@echo "    ab-compare      A/B compare current weights vs latest archived snapshot"
	@echo ""
	@echo "  Maintenance"
	@echo "    clean-cache     Remove .cache/ (forces FAISS index rebuild on next run)"

install: ## Install Python dependencies.
	$(PIP) install -r requirements.txt

verify: ## Check MySQL connectivity using .env.
	$(PYTHON) test_db_connection.py

run: ## Launch the Streamlit UI.
	streamlit run app.py

cli: ## Run a CLI search. Override SEARCH_QUERY=... for a different prompt.
	$(PYTHON) main.py "$${SEARCH_QUERY:-Senior technical recruiter hiring ML engineers}" --top-k 6

smoke: ## Headless end-to-end test across the prompt fixture.
	$(PYTHON) scripts/smoke_test.py

test: ## pytest unit + snapshot tests.
	$(PYTHON) -m pytest -q

calibrate: ## Dry-run the calibrator; print report only.
	$(PYTHON) scripts/calibrate.py --dry-run

calibrate-write: ## Fit and write config/weights.json + archive a history snapshot.
	$(PYTHON) scripts/calibrate.py

ablation: ## R4-4 ablation: 5 configs × 10 prompts, top-3 Jaccard vs full pipeline.
	$(PYTHON) scripts/ablation_table.py \
		--prompts fixtures/smoke_prompts.yml \
		--top-k 3

ab-compare: ## A/B compare current weights vs the most recent archived snapshot.
	$(PYTHON) scripts/ab_compare_weights.py \
		--weights-a config/weights.json \
		--weights-b "$$($(PYTHON) -c 'from pathlib import Path; from src.weights_loader import find_most_recent_archive; p=find_most_recent_archive(exclude_path=Path(\"config/weights.json\")); print(p or \"\")')" \
		--prompts fixtures/smoke_prompts.yml

seed-labels: ## Load 21 synthetic labels (labeler='seed-demo') for the calibration demo.
	$(PYTHON) scripts/load_seed_labels.py

check: install verify test smoke ## Full sanity sweep: install + verify + test + smoke.
	@echo ""
	@echo "All checks passed. You can now: make run"

clean-cache: ## Remove .cache/ (FAISS embedding cache).
	@rm -rf .cache/ || true
	@echo "Removed .cache/. Next run will rebuild the embedding index."
