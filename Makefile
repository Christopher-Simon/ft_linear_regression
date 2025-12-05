#///////////////////////////////////////////////////////////////////////////#
#																			#
#								VARIABLES									#
#																			#
#///////////////////////////////////////////////////////////////////////////#

PYTHON_SRC_DIR := src

VENV := .venv
PYTHON := python3
PIP := $(VENV)/bin/pip

#///////////////////////////////////////////////////////////////////////////#
#																			#
#								MANAGEMENT									#
#																			#
#///////////////////////////////////////////////////////////////////////////#

all: install run

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

run: $(VENV)/bin/activate
	$(VENV)/bin/python $(PYTHON_SRC_DIR)/main.py

lint: $(VENV)/bin/activate
	$(VENV)/bin/pylint src/

format: $(VENV)/bin/activate
	$(VENV)/bin/black src/

test: $(VENV)/bin/activate
	$(VENV)/bin/pytest tests/

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache

fclean: clean

venv_command: $(VENV)/bin/activate
	$(VENV)/bin/python $(CMD)

re: clean install

.PHONY: install run lint format test clean re
