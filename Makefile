#///////////////////////////////////////////////////////////////////////////#
#																			#
#								VARIABLES									#
#																			#
#///////////////////////////////////////////////////////////////////////////#


DOCKER_COMPOSE_FILE := docker-compose.yml
SERVICE_NAME := ft_linear_regression
CONTAINER_NAME := ft_linear_container
IMAGE_NAME := ft_linear_image
PYTHON_SRC_DIR := src

VENV := .venv
PYTHON := python3
PIP := $(VENV)/bin/pip

#///////////////////////////////////////////////////////////////////////////#
#																			#
#								MANAGEMENT									#
#																			#
#///////////////////////////////////////////////////////////////////////////#

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

install: $(VENV)/bin/activate requirements.txt
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

re: clean install

.PHONY: install run lint format test clean re
