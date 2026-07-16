set shell := ["zsh", "-cu"]

sync:
    uv sync --all-groups --extra torch --extra mlx --locked

fmt: sync
    uv run --no-sync ruff format .

check: sync
    uv run --no-sync ruff format --check .
    uv run --no-sync ruff check .
    uv run --no-sync ty check --error-on-warning src tests tools benchmarks

test: sync
    uv run --no-sync pytest -q

full: check test
    uv build

# Optional co-development path for an editable sibling engine checkout.
dev-check:
    python3 tools/dev.py -- ruff format --check .
    python3 tools/dev.py -- ruff check .
    python3 tools/dev.py -- ty check --error-on-warning src tests tools benchmarks

dev-test:
    python3 tools/dev.py -- pytest -q

bench-neural variant="nn_bin" rows="100000" validation_rows="10000" epochs="10" execution="profile" threads="1":
    uv run --no-sync python -m benchmarks.neural_training --variant {{variant}} --rows {{rows}} --validation-rows {{validation_rows}} --epochs {{epochs}} --execution {{execution}} --threads {{threads}}

bench-inference variant="nn_bin" rows="100000" batch_size="8192" execution="profile" threads="1":
    uv run --no-sync python -m benchmarks.neural_inference --variant {{variant}} --rows {{rows}} --batch-size {{batch_size}} --execution {{execution}} --threads {{threads}}
