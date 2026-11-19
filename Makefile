PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
PYTEST := $(PY) -m pytest
RUFF := $(PY) -m ruff

.PHONY: install run test lint format check clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

run:
	$(PY) main.py

test:
	$(PYTEST) -q

lint:
	$(RUFF) check --select E4,E7,E9,F,Q --ignore E402 main.py src tests

format:
	$(RUFF) format main.py src tests
	$(RUFF) check --select Q --fix main.py src tests

check: lint test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
