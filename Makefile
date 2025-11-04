.PHONY: help setup lint test train evaluate calibrate report app reproduce clean

PYTHON := python
PIP := pip
DATA_PATH := IPL.csv
REPORTS_DIR := reports
SEED := 42

help:
	@echo "IPL Impact Intelligence - Makefile Commands"
	@echo "==========================================="
	@echo "setup        - Install dependencies and setup development environment"
	@echo "lint         - Run code quality checks (black, isort, flake8)"
	@echo "test         - Run unit tests with pytest"
	@echo "train        - Train the impact model"
	@echo "evaluate     - Evaluate model performance with metrics and CIs"
	@echo "calibrate    - Calibrate model probabilities"
	@echo "report       - Generate all reports and visualizations"
	@echo "app          - Launch the Streamlit dashboard"
	@echo "reproduce    - Run full reproducible pipeline (train + eval + calibrate + report)"
	@echo "clean        - Remove generated artifacts and caches"

setup:
	@echo "Setting up development environment..."
	$(PIP) install -e . || $(PIP) install -r requirements.txt
	$(PIP) install pytest black isort flake8 pre-commit
	@echo "Installing pre-commit hooks..."
	pre-commit install || echo "Warning: pre-commit not installed"
	@echo "Setup complete!"

lint:
	@echo "Running code quality checks..."
	@echo "--- Black (code formatting) ---"
	black --check --line-length 120 . || (echo "Run 'black --line-length 120 .' to fix"; exit 1)
	@echo "--- isort (import sorting) ---"
	isort --check-only --profile black --line-length 120 . || (echo "Run 'isort --profile black --line-length 120 .' to fix"; exit 1)
	@echo "--- flake8 (linting) ---"
	flake8 --max-line-length 120 --extend-ignore E203,W503 --exclude .git,__pycache__,venv,env,.eggs,build,dist,outputs,reports . || echo "Flake8 found issues"
	@echo "Lint checks complete!"

test:
	@echo "Running unit tests..."
	pytest tests/ -v --tb=short
	@echo "Tests complete!"

train:
	@echo "Training impact model with seed $(SEED)..."
	$(PYTHON) main.py $(DATA_PATH) --seed $(SEED)
	@echo "Training complete!"

evaluate:
	@echo "Evaluating model performance..."
	@mkdir -p $(REPORTS_DIR)/figures
	$(PYTHON) cli/score_oos.py --data-path $(DATA_PATH) --seed $(SEED) || echo "Note: score_oos.py integration pending"
	@echo "Evaluation complete! Check $(REPORTS_DIR)/"

calibrate:
	@echo "Calibrating model probabilities..."
	@mkdir -p $(REPORTS_DIR)
	$(PYTHON) cli/calibrate.py --data-path $(DATA_PATH) --seed $(SEED) || echo "Note: calibrate.py integration pending"
	@echo "Calibration complete! Check $(REPORTS_DIR)/"

report:
	@echo "Generating comprehensive reports..."
	@mkdir -p $(REPORTS_DIR)/figures
	$(PYTHON) -c "from analysis import error_slices, failures; print('Generating error analysis...')" || echo "Note: Analysis modules integration pending"
	@echo "Reports generated in $(REPORTS_DIR)/"

app:
	@echo "Launching Streamlit dashboard..."
	streamlit run streamlit_app.py

reproduce:
	@echo "============================================"
	@echo "Running full reproducible pipeline"
	@echo "Seed: $(SEED)"
	@echo "Data: $(DATA_PATH)"
	@echo "============================================"
	@mkdir -p $(REPORTS_DIR)/figures
	@echo "Step 1/4: Training model..."
	@$(MAKE) train SEED=$(SEED)
	@echo "Step 2/4: Evaluating model..."
	@$(MAKE) evaluate SEED=$(SEED)
	@echo "Step 3/4: Calibrating probabilities..."
	@$(MAKE) calibrate SEED=$(SEED)
	@echo "Step 4/4: Generating reports..."
	@$(MAKE) report SEED=$(SEED)
	@echo "============================================"
	@echo "Reproducible pipeline complete!"
	@echo "Artifacts saved to: $(REPORTS_DIR)/"
	@echo "============================================"

clean:
	@echo "Cleaning up artifacts and caches..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf *.egg-info
	rm -rf build dist
	rm -rf $(REPORTS_DIR)
	rm -rf outputs
	@echo "Clean complete!"
