PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

.PHONY: install run test lint format check clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

run:
	$(VENV)/bin/python main.py

test:
	$(PYTEST) -q

lint:
	$(RUFF) check --select E4,E7,E9,F --ignore E402 main.py src tests

format:
	$(RUFF) format main.py src tests

check: lint test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
