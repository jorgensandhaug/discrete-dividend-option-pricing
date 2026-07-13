set shell := ["zsh", "-cu"]

sync:
    uv sync --all-groups --locked

fmt:
    python3 tools/dev.py -- ruff format .

check:
    python3 tools/dev.py -- ruff format --check .
    python3 tools/dev.py -- ruff check .
    python3 tools/dev.py -- mypy

test:
    python3 tools/dev.py -- pytest -q

full: check test
    uv build

