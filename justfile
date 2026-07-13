set shell := ["zsh", "-cu"]

sync:
    uv sync --all-groups --extra torch --extra mlx --locked

fmt:
    python3 tools/dev.py -- ruff format .

engine-revision:
    test "$(tr -d '[:space:]' < config/engine-revision.txt)" = "$(git -C ../finoptions-engine rev-parse HEAD)"

check: engine-revision
    python3 tools/dev.py -- ruff format --check .
    python3 tools/dev.py -- ruff check .
    python3 tools/dev.py -- ty check --error-on-warning src tests tools

test:
    python3 tools/dev.py -- pytest -q

full: check test
    uv build
