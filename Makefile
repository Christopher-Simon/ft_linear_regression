#///////////////////////////////////////////////////////////////////////////#
#                               VARIABLES                                   #
#///////////////////////////////////////////////////////////////////////////#

PYTHON_SRC_DIR := src

#///////////////////////////////////////////////////////////////////////////#
#                               MANAGEMENT                                  #
#///////////////////////////////////////////////////////////////////////////#

all: dev run

dev:
	uv sync

run:
	uv run python $(PYTHON_SRC_DIR)/main.py

check:
	uv run ruff check --fix src/

format:
	uv run ruff format src/

typecheck:
	uv run mypy src/

lint:
	@echo "Starting linting..." > lint.log
	@echo "--- Ruff Check ---" >> lint.log
	-uv run ruff check --fix src/ >> lint.log 2>&1
	@echo "\n--- Ruff Format ---" >> lint.log
	-uv run ruff format src/ >> lint.log 2>&1
	@echo "\n--- Mypy ---" >> lint.log
	-uv run mypy src/ >> lint.log 2>&1
	@echo "Linting complete. Check 'lint.log' for details."

test:
	uv run pytest -s -vv src/tests/

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

fclean: clean

re: clean dev

.PHONY: dev run lint format typecheck test clean fclean re