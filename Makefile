VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
UV = $(VENV_DIR)/bin/uv

all: setup

$(VENV_DIR)/bin/activate: pyproject.toml
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV_DIR)
	@echo "Bootstrapping 'uv' via standard pip..."
	@$(PIP) install --upgrade pip uv
	@echo "Using 'uv' to install project dependencies ultra-fast..."
	@$(UV) sync --all-groups

setup: $(VENV_DIR)/bin/activate


train: setup
	@echo "Running Training Program..."
	@$(UV) run train.py $(ARGS)

predict: setup
	@echo "Running Prediction Program..."
	@$(UV) run predict.py $(ARGS)

evaluate: setup
	@echo "Running Evaluation Program..."
	@$(UV) run evaluate.py $(ARGS)

test:
	@echo "Running Evaluation Program..."
	@$(UV) run pytest

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
