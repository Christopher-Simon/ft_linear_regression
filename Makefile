# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
UV = $(VENV_DIR)/bin/uv

# The default rule
all: setup

# Create the venv, install uv with pip, then use uv for the heavy lifting!
$(VENV_DIR)/bin/activate: pyproject.toml
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV_DIR)
	@echo "Bootstrapping 'uv' via standard pip..."
	@$(PIP) install --upgrade pip uv
	@echo "Using 'uv' to install project dependencies ultra-fast..."
	@$(UV) pip install .
	@touch $(VENV_DIR)/bin/activate

# Alias for the setup process
setup: $(VENV_DIR)/bin/activate

# --- Executable Targets ---

train: setup
	@echo "Running Training Program..."
	@$(PYTHON) train.py $(ARGS)

predict: setup
	@echo "Running Prediction Program..."
	@$(PYTHON) predict.py $(ARGS)

evaluate: setup
	@echo "Running Evaluation Program..."
	@$(PYTHON) evaluate.py $(ARGS)

# --- Cleanup Targets ---


clean:
	@echo "Cleaning cache and virtual environment..."
	@rm -rf $(VENV_DIR)
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type d -name "*_cache" -exec rm -r {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true

fclean: clean
	@echo "Removing model weights..."
	@rm -f model_weights.json
	@rm -f model/model_weights.json

re: fclean all

.PHONY: all setup train predict evaluate clean fclean re
