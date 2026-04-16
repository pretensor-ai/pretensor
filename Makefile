# Dev workflow: prefers `uv` when installed; otherwise uses a local `.venv`
# (create with `python3 -m venv .venv` if `ensurepip` is available).
UV ?= uv
VENV ?= .venv
PY = $(VENV)/bin/python
HAS_UV := $(shell command -v $(UV) >/dev/null 2>&1 && echo yes)
HAS_VENV_PY := $(shell test -x $(VENV)/bin/python && echo yes)

ifeq ($(HAS_UV),yes)
RUN = $(UV) run
else ifeq ($(HAS_VENV_PY),yes)
RUN = $(PY) -m
else
# Fall back to system Python when neither uv nor `.venv` is present (requires dev deps on PATH).
RUN = python3 -m
endif

.PHONY: install test test-e2e lint typecheck format verify smoke-quickstart

install:
ifeq ($(HAS_UV),yes)
	$(UV) sync --extra dev
	$(UV) run pre-commit install
	$(UV) run pre-commit install --hook-type commit-msg
else
	@test -d $(VENV) || { echo "Create $(VENV) first: python3 -m venv $(VENV)"; exit 1; }
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pre_commit install
	$(PY) -m pre_commit install --hook-type commit-msg
endif

test:
	PYTHONPATH=src:. $(RUN) pytest tests/

test-e2e:
	PRETENSOR_E2E=1 $(RUN) pytest tests/e2e/ -v --tb=short

lint:
	$(RUN) ruff check src tests

typecheck:
	$(RUN) pyright

format:
	$(RUN) ruff format src tests

# End-to-end smoke for `pretensor quickstart`: runs docker compose up + index,
# then parses the printed mcpServers JSON block, then tears down. Opt-in —
# requires docker; not wired into CI.
smoke-quickstart:
	@command -v docker >/dev/null 2>&1 || { echo "SKIP: docker not on PATH"; exit 0; }
	@tmp=$$(mktemp -d); \
	trap "$(RUN) pretensor quickstart --down >/dev/null 2>&1 || true; rm -rf $$tmp" EXIT; \
	echo "smoke: quickstart → $$tmp"; \
	$(RUN) pretensor quickstart --state-dir $$tmp 2> $$tmp/stderr.log; \
	status=$$?; \
	if [ $$status -ne 0 ]; then echo "FAIL: quickstart exit $$status"; tail -20 $$tmp/stderr.log; exit $$status; fi; \
	python3 -c "import json, re, sys; s = open('$$tmp/stderr.log').read(); m = re.search(r'\{[^{}]*\"mcpServers\"[\\s\\S]*\\}', s); assert m, 'mcpServers JSON not found'; json.loads(m.group(0)); print('OK: mcpServers JSON parsed')"

# Run before every commit: same checks as CI (ruff check, pyright, pytest). Requires dev deps (`make install` or `uv sync --extra dev`).
verify:
ifeq ($(HAS_UV),yes)
	$(UV) sync --extra dev
endif
	$(RUN) ruff check src tests
	$(RUN) pyright
	PYTHONPATH=src:. $(RUN) pytest tests/
